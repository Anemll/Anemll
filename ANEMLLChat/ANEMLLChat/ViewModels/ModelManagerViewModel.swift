//
//  ModelManagerViewModel.swift
//  ANEMLLChat
//
//  ViewModel for model management
//

import Foundation
import SwiftUI
import Observation

/// View model for managing models
@Observable
@MainActor
final class ModelManagerViewModel {
    // MARK: - State

    /// All available models
    var availableModels: [ModelInfo] = []

    /// Currently downloading model ID
    var downloadingModelId: String?

    /// Current download progress
    var downloadProgress: DownloadProgress?

    /// Currently loaded model ID
    var loadedModelId: String?

    /// Model loading progress
    var loadingProgress: ModelLoadingProgress?

    /// Error message
    var errorMessage: String?

    /// Whether model is being loaded
    var isLoadingModel: Bool = false

    /// ID of model currently being loaded
    var loadingModelId: String?

    // MARK: - Computed Properties

    /// Downloaded models
    var downloadedModels: [ModelInfo] {
        availableModels.filter { $0.isDownloaded }
    }

    /// Models available for download (excludes errored models)
    var availableForDownload: [ModelInfo] {
        availableModels.filter { !$0.isDownloaded && !$0.isDownloading && $0.downloadError == nil }
    }

    /// Total size of downloaded models
    var downloadedModelsSize: String {
        let total = downloadedModels.compactMap { $0.sizeBytes }.reduce(0, +)
        return ByteCountFormatter.string(fromByteCount: total, countStyle: .file)
    }

    // MARK: - Initialization

    init() {
        // Immediately show default models (before async check completes)
        availableModels = ModelInfo.defaultModels
        logInfo("ModelManagerViewModel init: set \(availableModels.count) default models", category: .model)

        // Then async check download status and load custom models
        Task {
            await loadModels()
        }
    }

    // MARK: - Model Loading

    /// Load model list (defaults + custom)
    func loadModels() async {
        logInfo("Loading models...", category: .model)

        // Start with defaults - these should ALWAYS be available
        var models = ModelInfo.defaultModels
        logDebug("Starting with \(models.count) default models", category: .model)

        // Add custom models (if any)
        do {
            let customModels = try await StorageService.shared.loadModelsRegistry()
            for custom in customModels {
                if !models.contains(where: { $0.id == custom.id }) {
                    models.append(custom)
                    logDebug("Added custom model: \(custom.id)", category: .model)
                }
            }
        } catch {
            logWarning("Failed to load custom models: \(error)", category: .model)
            // Continue with defaults only
        }

        // Check download status and reset stale download state
        for i in models.indices {
            let isDownloaded = await StorageService.shared.isModelDownloaded(models[i].id)
            models[i].isDownloaded = isDownloaded

            // Reset stale downloading state from previous sessions
            // (downloads don't survive app restart)
            models[i].isDownloading = false
            models[i].downloadProgress = nil
            models[i].downloadError = nil

            if isDownloaded {
                let path = await StorageService.shared.modelPath(for: models[i].id)
                models[i].localPath = path.path
                models[i].metaYamlPath = path.appendingPathComponent("meta.yaml").path
                logDebug("Model \(models[i].name) is downloaded at \(path.path)", category: .model)
            }
        }

        // Update the published property - this triggers UI update
        availableModels = models
        logInfo("Loaded \(models.count) models (\(downloadedModels.count) downloaded)", category: .model)

        // Auto-load last model after models are loaded (if setting enabled)
        if await StorageService.shared.autoLoadLastModel {
            await autoLoadLastModel()
        }
    }

    /// Refresh download status for all models
    func refreshModelStatus() async {
        for i in availableModels.indices {
            let isDownloaded = await StorageService.shared.isModelDownloaded(availableModels[i].id)
            availableModels[i].isDownloaded = isDownloaded

            if isDownloaded {
                let path = await StorageService.shared.modelPath(for: availableModels[i].id)
                availableModels[i].localPath = path.path
                availableModels[i].metaYamlPath = path.appendingPathComponent("meta.yaml").path
            }
        }
    }

    // MARK: - Download

    /// Download a model (public API - sets state and starts download)
    func downloadModel(_ model: ModelInfo) async {
        guard !model.isDownloaded, !model.isDownloading else { return }

        // Set state immediately so UI updates
        downloadingModelId = model.id
        updateModelDownloading(model.id, isDownloading: true)

        // Perform the actual download
        await performDownload(model)
    }

    /// Cancel ongoing download
    func cancelDownload() async {
        guard let modelId = downloadingModelId else { return }

        await DownloadService.shared.cancelDownload(modelId)
        updateModelDownloading(modelId, isDownloading: false)

        downloadingModelId = nil
        downloadProgress = nil

        logInfo("Download cancelled: \(modelId)", category: .download)
    }

    /// Delete a downloaded model
    func deleteModel(_ model: ModelInfo) async {
        do {
            try await StorageService.shared.deleteModel(model.id)

            if let index = availableModels.firstIndex(where: { $0.id == model.id }) {
                availableModels[index].isDownloaded = false
                availableModels[index].localPath = nil
                availableModels[index].metaYamlPath = nil
            }

            // Unload if currently loaded
            if loadedModelId == model.id {
                loadedModelId = nil
            }

            logInfo("Deleted model: \(model.id)", category: .model)

        } catch {
            errorMessage = error.localizedDescription
            logError("Failed to delete model: \(error)", category: .model)
        }
    }

    // MARK: - Model Loading

    /// Load a model for inference
    func loadModelForInference(_ model: ModelInfo) async {
        guard model.isDownloaded, let path = model.localPath else {
            errorMessage = "Model not downloaded"
            return
        }

        isLoadingModel = true
        loadingModelId = model.id
        loadingProgress = nil
        errorMessage = nil

        // Start a task to poll InferenceService's loading progress
        let progressTask = Task { @MainActor in
            while !Task.isCancelled && isLoadingModel {
                loadingProgress = InferenceService.shared.loadingProgress
                try? await Task.sleep(for: .milliseconds(100))
            }
        }

        do {
            let modelURL = URL(fileURLWithPath: path)
            try await InferenceService.shared.loadModel(from: modelURL)
            loadedModelId = model.id

            // Save as selected model
            await StorageService.shared.saveSelectedModelId(model.id)

            logInfo("Model loaded: \(model.id)", category: .model)

        } catch {
            errorMessage = error.localizedDescription
            logError("Failed to load model: \(error)", category: .model)
        }

        progressTask.cancel()
        isLoadingModel = false
        loadingModelId = nil
        loadingProgress = nil
    }

    /// Unload the current model
    func unloadCurrentModel() async {
        await InferenceService.shared.unloadModel()
        loadedModelId = nil
        logInfo("Model unloaded", category: .model)
    }

    /// Auto-load the last selected model
    func autoLoadLastModel() async {
        guard let selectedId = await StorageService.shared.selectedModelId else {
            logInfo("[AUTO-LOAD] No saved model ID found", category: .model)
            return
        }

        logInfo("[AUTO-LOAD] Looking for model: \(selectedId)", category: .model)

        if let model = availableModels.first(where: { $0.id == selectedId && $0.isDownloaded }) {
            logInfo("[AUTO-LOAD] Found model, loading: \(model.name)", category: .model)
            await loadModelForInference(model)
        } else {
            logWarning("[AUTO-LOAD] Model not found or not downloaded: \(selectedId)", category: .model)
        }
    }

    // MARK: - Custom Models

    /// Add a custom model from URL and start download
    /// NOTE: This returns immediately after adding the model - download runs in background
    func addCustomModel(repoId: String, name: String) async {
        // Trim whitespace from inputs to prevent path issues
        let cleanRepoId = repoId.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanName = name.trimmingCharacters(in: .whitespacesAndNewlines)

        logInfo("addCustomModel called: '\(cleanRepoId)'", category: .model)

        // Validate repo ID format
        guard !cleanRepoId.isEmpty, cleanRepoId.contains("/") else {
            errorMessage = "Invalid repository ID format. Use: owner/repo-name"
            logError("Invalid repo ID format: '\(cleanRepoId)'", category: .model)
            return
        }

        // Check if model already exists
        if let existingIndex = availableModels.firstIndex(where: { $0.id == cleanRepoId }) {
            let existingModel = availableModels[existingIndex]

            // If already downloaded, just inform user
            if existingModel.isDownloaded {
                errorMessage = "Model already downloaded: \(cleanName)"
                logInfo("Model already downloaded: \(cleanRepoId)", category: .model)
                return
            }

            // If currently downloading, don't start another
            if existingModel.isDownloading {
                logInfo("Model already downloading: \(cleanRepoId)", category: .model)
                return
            }

            // Model exists but not downloaded - start download
            logInfo("Starting download for existing model: \(cleanRepoId)", category: .model)

            // Mark as downloading IMMEDIATELY so UI shows it
            downloadingModelId = existingModel.id
            updateModelDownloading(existingModel.id, isDownloading: true)

            Task {
                await performDownload(existingModel)
            }
            return
        }

        // New model - add to list
        let model = ModelInfo(
            id: cleanRepoId,
            name: cleanName.isEmpty ? cleanRepoId.components(separatedBy: "/").last ?? cleanRepoId : cleanName,
            description: "Custom model from HuggingFace",
            size: "Unknown"
        )

        availableModels.append(model)
        logInfo("Added model to list: \(cleanRepoId), total models: \(availableModels.count)", category: .model)

        // Save to registry FIRST (before download starts)
        do {
            try await StorageService.shared.saveModelsRegistry(availableModels)
            logInfo("Saved model registry with \(availableModels.count) models", category: .model)
        } catch {
            logError("Failed to save model registry: \(error)", category: .model)
        }

        // Mark as downloading IMMEDIATELY so UI shows it in Downloading section
        // This must happen BEFORE the async Task to avoid UI timing gap
        downloadingModelId = model.id
        updateModelDownloading(model.id, isDownloading: true)

        // Start download in background (don't await - return immediately)
        Task {
            await performDownload(model)
        }
    }

    /// Internal download implementation (called after downloadingModelId is set)
    private func performDownload(_ model: ModelInfo) async {
        logInfo("Starting download: \(model.id)", category: .download)

        await DownloadService.shared.downloadModel(
            model.id,
            progress: { [weak self] progress in
                Task { @MainActor in
                    self?.downloadProgress = progress
                    self?.updateModelProgress(model.id, progress: progress)
                }
            },
            completion: { [weak self] result in
                Task { @MainActor in
                    guard let self = self else { return }

                    switch result {
                    case .success(let path):
                        self.updateModelDownloaded(model.id, path: path)
                        logInfo("Download complete: \(model.id)", category: .download)

                    case .failure(let error):
                        self.updateModelError(model.id, error: error.localizedDescription)
                        self.errorMessage = error.localizedDescription
                        logError("Download failed: \(error)", category: .download)
                    }

                    self.downloadingModelId = nil
                    self.downloadProgress = nil
                }
            }
        )
    }

    // MARK: - Helpers

    private func updateModelDownloading(_ id: String, isDownloading: Bool) {
        if let index = availableModels.firstIndex(where: { $0.id == id }) {
            availableModels[index].isDownloading = isDownloading
            if !isDownloading {
                availableModels[index].downloadProgress = nil
                availableModels[index].downloadError = nil
            }
        }
    }

    private func updateModelProgress(_ id: String, progress: DownloadProgress) {
        if let index = availableModels.firstIndex(where: { $0.id == id }) {
            availableModels[index].downloadProgress = progress.progress
            availableModels[index].downloadedBytes = progress.downloadedBytes
        }
    }

    private func updateModelDownloaded(_ id: String, path: URL) {
        if let index = availableModels.firstIndex(where: { $0.id == id }) {
            availableModels[index].isDownloaded = true
            availableModels[index].isDownloading = false
            availableModels[index].localPath = path.path
            availableModels[index].metaYamlPath = path.appendingPathComponent("meta.yaml").path
            availableModels[index].downloadProgress = nil
            availableModels[index].downloadError = nil
        }
    }

    private func updateModelError(_ id: String, error: String) {
        if let index = availableModels.firstIndex(where: { $0.id == id }) {
            availableModels[index].isDownloading = false
            availableModels[index].downloadError = error
        }
    }
}
