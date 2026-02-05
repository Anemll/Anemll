//
//  ModelManagerViewModel.swift
//  ANEMLLChat
//
//  ViewModel for model management
//

import Foundation
import SwiftUI
import Observation
#if os(iOS)
import UIKit
#endif

// MARK: - Device Type Detection

/// Device type for determining weight file size limits
enum DeviceType {
    case mac
    case macCatalyst
    case iPad
    case iPhone
    case other

    static var current: DeviceType {
        #if os(macOS)
        return .mac
        #elseif targetEnvironment(macCatalyst)
        return .macCatalyst
        #else
        let device = UIDevice.current
        if device.userInterfaceIdiom == .pad {
            return .iPad
        } else if device.userInterfaceIdiom == .phone {
            return .iPhone
        } else {
            return .other
        }
        #endif
    }

    /// Check if the device has an M-series chip (Apple Silicon)
    static var hasMSeriesChip: Bool {
        #if os(macOS)
        var sysinfo = utsname()
        uname(&sysinfo)
        let machine = withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        }
        return machine?.contains("arm64") ?? false
        #elseif targetEnvironment(macCatalyst)
        return true
        #else
        var sysinfo = utsname()
        uname(&sysinfo)
        let machine = withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        }
        guard let identifier = machine else { return false }

        // iPads with M-series chips
        let mSeriesIPads = [
            "iPad13,4", "iPad13,5", "iPad13,6", "iPad13,7",  // iPad Pro 11" 3rd gen (M1)
            "iPad13,8", "iPad13,9", "iPad13,10", "iPad13,11", // iPad Pro 12.9" 5th gen (M1)
            "iPad13,16", "iPad13,17",                         // iPad Air 5th gen (M1)
            "iPad14,3", "iPad14,4",                           // iPad Pro 11" 4th gen (M2)
            "iPad14,5", "iPad14,6",                           // iPad Pro 12.9" 6th gen (M2)
            "iPad14,8", "iPad14,9",                           // iPad Air 6th gen (M2)
            "iPad16,3", "iPad16,4",                           // iPad Pro 11" 5th gen (M4)
            "iPad16,5", "iPad16,6",                           // iPad Pro 13" 1st gen (M4)
        ]
        return mSeriesIPads.contains(identifier)
        #endif
    }

    /// Check if device requires weight file size limit (1GB per weight file)
    static var requiresWeightSizeLimit: Bool {
        switch current {
        case .iPhone:
            return true
        case .iPad:
            return !hasMSeriesChip
        case .mac, .macCatalyst:
            return false
        case .other:
            return true
        }
    }

    /// Maximum weight file size in bytes (1GB for limited devices)
    static let maxWeightFileSize: Int64 = 1_073_741_824  // 1 GB
}

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

    /// Warning message shown before loading a model with oversized weights
    var weightWarningMessage: String?

    /// Whether to show the weight warning alert
    var showWeightWarningAlert: Bool = false

    /// Model pending load (waiting for user confirmation)
    private var pendingLoadModel: ModelInfo?

    /// Load a model for inference
    func loadModelForInference(_ model: ModelInfo) async {
        guard model.isDownloaded, let path = model.localPath else {
            errorMessage = "Model not downloaded"
            return
        }

        // Check for weight size warning
        if let warning = getWeightSizeWarning(for: model) {
            weightWarningMessage = warning
            pendingLoadModel = model
            showWeightWarningAlert = true
            logWarning("Model has weight size warning: \(warning)", category: .model)
            return
        }

        await performModelLoad(model, path: path)
    }

    /// Continue loading model after user confirmed weight warning
    func confirmLoadModel() async {
        guard let model = pendingLoadModel, let path = model.localPath else {
            pendingLoadModel = nil
            showWeightWarningAlert = false
            return
        }

        showWeightWarningAlert = false
        weightWarningMessage = nil
        pendingLoadModel = nil

        await performModelLoad(model, path: path)
    }

    /// Cancel model load
    func cancelLoadModel() {
        pendingLoadModel = nil
        showWeightWarningAlert = false
        weightWarningMessage = nil
    }

    /// Internal method to perform model loading
    private func performModelLoad(_ model: ModelInfo, path: String) async {
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

    // MARK: - Weight File Size Checking

    /// Get detailed weight file information for a model
    /// - Parameter model: The model to check
    /// - Returns: Tuple with (largestWeightSize, largestWeightName, allWeightFiles) or nil if not available
    func getWeightFileDetails(for model: ModelInfo) -> (largest: Int64, largestName: String, files: [(name: String, size: Int64)])? {
        guard let localPath = model.localPath else { return nil }

        let modelDir = URL(fileURLWithPath: localPath)
        let fileManager = FileManager.default

        guard fileManager.fileExists(atPath: modelDir.path) else { return nil }

        var weightFiles: [(name: String, size: Int64)] = []

        // Check all .mlmodelc directories for weight.bin files
        do {
            let contents = try fileManager.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            let mlmodelcDirs = contents.filter { $0.pathExtension == "mlmodelc" }

            for mlmodelcDir in mlmodelcDirs {
                let dirName = mlmodelcDir.lastPathComponent

                // Check both possible weight file locations
                let weightPaths = [
                    mlmodelcDir.appendingPathComponent("weights/weight.bin"),
                    mlmodelcDir.appendingPathComponent("weight.bin")
                ]

                for weightPath in weightPaths {
                    if fileManager.fileExists(atPath: weightPath.path) {
                        if let attrs = try? fileManager.attributesOfItem(atPath: weightPath.path),
                           let size = attrs[.size] as? Int64, size > 0 {
                            weightFiles.append((name: dirName, size: size))
                        }
                        break
                    }
                }
            }
        } catch {
            logError("Error getting weight file details: \(error)", category: .model)
            return nil
        }

        guard !weightFiles.isEmpty else { return nil }

        let sorted = weightFiles.sorted { $0.size > $1.size }
        let largest = sorted.first!

        return (largest: largest.size, largestName: largest.name, files: weightFiles)
    }

    /// Get warning message if weight files exceed 1GB on limited devices
    /// - Parameter model: The model to check
    /// - Returns: Warning message or nil if no issue
    func getWeightSizeWarning(for model: ModelInfo) -> String? {
        guard DeviceType.requiresWeightSizeLimit else { return nil }
        guard let details = getWeightFileDetails(for: model) else { return nil }

        let oversizedFiles = details.files.filter { $0.size > DeviceType.maxWeightFileSize }
        guard !oversizedFiles.isEmpty else { return nil }

        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file

        let fileList = oversizedFiles.map { "\($0.name): \(formatter.string(fromByteCount: $0.size))" }.joined(separator: ", ")
        let deviceName: String
        switch DeviceType.current {
        case .iPhone:
            deviceName = "iPhone"
        case .iPad:
            deviceName = "this iPad (non-M-series)"
        default:
            deviceName = "this device"
        }

        return "Model may not load on \(deviceName). Weight file(s) exceed 1GB limit: \(fileList)"
    }

    /// Check if model has any weight files exceeding 1GB (regardless of device)
    /// - Parameter model: The model to check
    /// - Returns: True if any weight file exceeds 1GB
    func hasOversizedWeights(for model: ModelInfo) -> Bool {
        guard let details = getWeightFileDetails(for: model) else { return false }
        return details.largest > DeviceType.maxWeightFileSize
    }
}
