//
//  ModelManagerViewModel.swift
//  ANEMLLChat
//
//  ViewModel for model management
//

import Foundation
import SwiftUI
import Observation
import CryptoKit
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
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

enum LocalModelImportMode: String, CaseIterable, Sendable {
    case importCopy
    case linkExternal

    var displayTitle: String {
        switch self {
        case .importCopy: return "Import (Copy)"
        case .linkExternal: return "Link (External)"
        }
    }
}

struct LocalModelInspection: Sendable {
    let droppedURL: URL
    let modelRootURL: URL
    let suggestedDisplayName: String
    let suggestedModelId: String
}

enum LocalModelValidationError: LocalizedError {
    case notDirectory
    case invalidStructure([String])

    var errorDescription: String? {
        switch self {
        case .notDirectory:
            return "Please drop a folder, not a file."
        case .invalidStructure(let issues):
            if issues.isEmpty {
                return "Folder does not look like a valid model root."
            }
            return issues.joined(separator: " ")
        }
    }
}

enum ModelPackageImportError: LocalizedError {
    case unsupportedPlatform
    case invalidPackageRoot
    case missingManifest
    case invalidManifest
    case unsupportedFormatVersion(Int)
    case incompatibleAppVersion(minimum: String, current: String)
    case missingFile(path: String)
    case fileHashMismatch(path: String)
    case invalidModelRoot

    var errorDescription: String? {
        switch self {
        case .unsupportedPlatform:
            return "Model package import is supported on iOS only."
        case .invalidPackageRoot:
            return "Could not resolve model package folder."
        case .missingManifest:
            return "Package is missing manifest.json."
        case .invalidManifest:
            return "Package manifest is invalid."
        case .unsupportedFormatVersion(let version):
            return "Unsupported package format version: \(version)."
        case .incompatibleAppVersion(let minimum, let current):
            return "Package requires app version \(minimum)+. Current version is \(current)."
        case .missingFile(let path):
            return "Package file is missing: \(path)."
        case .fileHashMismatch(let path):
            return "Hash validation failed for \(path)."
        case .invalidModelRoot:
            return "Package does not contain a valid model root."
        }
    }
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

    /// Recently added model ID (used by UI to highlight new entries)
    var justAddedModelId: String?

    /// Model loading progress
    var loadingProgress: ModelLoadingProgress?

    /// Error message
    var errorMessage: String?

    #if os(iOS)
    /// Non-error status shown while/after importing an incoming package.
    var incomingTransferStatusMessage: String?

    /// Indicates that a package import is currently running.
    var isImportingIncomingPackage: Bool = false

    /// Signals UI to focus/open model list after a package import.
    var lastImportedPackageModelId: String?

    private var recentlyHandledIncomingURLKeys: [String: Date] = [:]
    #endif

    /// Whether model is being loaded
    var isLoadingModel: Bool = false

    /// ID of model currently being loaded
    var loadingModelId: String?

    private var clearJustAddedTask: Task<Void, Never>?
    #if os(macOS)
    private var preparedTransferPackages: [URL] = []
    #endif

    // MARK: - Computed Properties

    /// Downloaded models
    var downloadedModels: [ModelInfo] {
        availableModels.filter { $0.isDownloaded }
    }

    /// Models available for download (excludes errored models)
    var availableForDownload: [ModelInfo] {
        availableModels.filter {
            $0.sourceKind == .huggingFace &&
            !$0.isDownloaded &&
            !$0.isDownloading &&
            $0.downloadError == nil
        }
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

        // Check model availability and reset stale download state
        for i in models.indices {
            models[i] = await refreshedModelStatus(for: models[i])
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
        var refreshedModels: [ModelInfo] = []
        refreshedModels.reserveCapacity(availableModels.count)

        for model in availableModels {
            refreshedModels.append(await refreshedModelStatus(for: model))
        }
        availableModels = refreshedModels
    }

    // MARK: - Download

    /// Download a model (public API - sets state and starts download)
    func downloadModel(_ model: ModelInfo) async {
        guard model.sourceKind == .huggingFace else {
            errorMessage = "Only HuggingFace models support download."
            return
        }
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
            if model.sourceKind != .localLinked {
                try await StorageService.shared.deleteModel(model.id)
            }

            if let index = availableModels.firstIndex(where: { $0.id == model.id }) {
                // Local-only models should be removed from list on delete.
                if model.sourceKind == .localImported || model.sourceKind == .localLinked {
                    availableModels.remove(at: index)
                } else {
                    availableModels[index].isDownloaded = false
                    availableModels[index].localPath = nil
                    availableModels[index].metaYamlPath = nil
                }
            }

            try? await StorageService.shared.saveModelsRegistry(availableModels)

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
            if model.sourceKind == .localLinked {
                errorMessage = "Linked source folder is unavailable. Re-link or re-import this model."
            } else {
                errorMessage = "Model not downloaded"
            }
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
            markModelAsJustAdded(existingModel.id)

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
        markModelAsJustAdded(model.id)
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

    // MARK: - Local Models (macOS import/link workflow)

    /// Inspect a dropped/selected folder and derive a deterministic local model proposal.
    func inspectLocalModelFolder(_ droppedURL: URL) throws -> LocalModelInspection {
        let modelRootURL = try detectLocalModelRoot(from: droppedURL)
        let baseName = suggestModelName(from: droppedURL)
        let uniqueName = uniqueDisplayName(for: baseName)
        let uniqueModelId = uniqueLocalModelId(forDisplayName: uniqueName)

        return LocalModelInspection(
            droppedURL: droppedURL,
            modelRootURL: modelRootURL,
            suggestedDisplayName: uniqueName,
            suggestedModelId: uniqueModelId
        )
    }

    /// Add a local model either by copying into app storage or linking the external folder.
    func addLocalModel(from droppedURL: URL, displayName: String, mode: LocalModelImportMode) async {
        errorMessage = nil

        do {
            let inspection = try inspectLocalModelFolder(droppedURL)
            let requestedName = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
            let finalName = uniqueDisplayName(for: requestedName.isEmpty ? inspection.suggestedDisplayName : requestedName)
            let finalModelId = uniqueLocalModelId(forDisplayName: finalName)
            let rootPath = inspection.modelRootURL.path

            if availableModels.contains(where: { $0.localPath == rootPath || $0.linkedPath == rootPath }) {
                errorMessage = "This model folder is already added."
                return
            }

            let localPath: String
            let linkedPath: String?
            let bookmarkDataBase64: String?
            let description: String

            switch mode {
            case .importCopy:
                let importedPath = try await StorageService.shared.importModelDirectory(from: inspection.modelRootURL, toModelId: finalModelId)
                localPath = importedPath.path
                linkedPath = nil
                bookmarkDataBase64 = nil
                description = "Local model (imported)"

            case .linkExternal:
                localPath = inspection.modelRootURL.path
                linkedPath = inspection.modelRootURL.path
                bookmarkDataBase64 = makeBookmarkDataBase64(for: inspection.modelRootURL)
                description = "Local model (linked)"
            }

            let newModel = ModelInfo(
                id: finalModelId,
                name: finalName,
                description: description,
                size: "Local",
                isDownloaded: true,
                localPath: localPath,
                metaYamlPath: URL(fileURLWithPath: localPath).appendingPathComponent("meta.yaml").path,
                sourceKind: mode == .importCopy ? .localImported : .localLinked,
                linkedPath: linkedPath,
                bookmarkDataBase64: bookmarkDataBase64
            )

            availableModels.append(newModel)
            markModelAsJustAdded(newModel.id)
            try await StorageService.shared.saveModelsRegistry(availableModels)
            logInfo("Added local model: \(newModel.id) (\(mode.rawValue))", category: .model)

        } catch {
            errorMessage = error.localizedDescription
            logError("Failed to add local model: \(error)", category: .model)
        }
    }

    // MARK: - Package Import (iOS receiver)

    func handleIncomingTransferURL(_ url: URL) async {
        #if os(iOS)
        guard shouldProcessIncomingTransferURL(url) else {
            logDebug("Ignoring duplicate incoming transfer URL: \(url.path)", category: .model)
            return
        }

        isImportingIncomingPackage = true
        incomingTransferStatusMessage = "Importing model package..."

        do {
            let imported = try await importModelPackage(from: url)
            incomingTransferStatusMessage = "Imported model: \(imported.name)"
            lastImportedPackageModelId = imported.id
            logInfo("Imported transferred model package: \(imported.id)", category: .model)
        } catch {
            errorMessage = error.localizedDescription
            logError("Failed to import transferred model package: \(error)", category: .model)
        }
        isImportingIncomingPackage = false
        #else
        logInfo("Ignoring incoming transfer URL on non-iOS platform: \(url.path)", category: .model)
        #endif
    }

    #if os(iOS)
    private func importModelPackage(from incomingURL: URL) async throws -> ModelInfo {
        let accessStarted = incomingURL.startAccessingSecurityScopedResource()
        defer {
            if accessStarted {
                incomingURL.stopAccessingSecurityScopedResource()
            }
        }

        let packageRoot = try resolveIncomingPackageRoot(from: incomingURL)
        let manifestURL = packageRoot.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelPackageImportError.missingManifest
        }

        let manifestData = try Data(contentsOf: manifestURL)
        guard let manifest = try? JSONDecoder().decode(ModelPackageManifest.self, from: manifestData) else {
            throw ModelPackageImportError.invalidManifest
        }

        try validatePackageCompatibility(manifest)
        try await validatePackageFiles(manifest: manifest, packageRoot: packageRoot)

        let modelRootURL: URL
        if let rootPath = manifest.modelRootPath, !rootPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            guard let normalizedRoot = Self.normalizePackageRelativePath(rootPath) else {
                throw ModelPackageImportError.invalidManifest
            }
            modelRootURL = packageRoot.appendingPathComponent(normalizedRoot, isDirectory: true)
        } else {
            modelRootURL = packageRoot
        }

        let resolvedModelRoot: URL
        do {
            resolvedModelRoot = try detectLocalModelRoot(from: modelRootURL)
        } catch {
            throw ModelPackageImportError.invalidModelRoot
        }

        let manifestName = manifest.modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        let preferredName = manifestName.isEmpty ? suggestModelName(from: resolvedModelRoot) : manifestName
        let finalName = uniqueDisplayName(for: preferredName)
        let finalModelId = uniqueLocalModelId(forDisplayName: finalName)

        let importedPath = try await StorageService.shared.importModelDirectory(from: resolvedModelRoot, toModelId: finalModelId)
        let newModel = ModelInfo(
            id: finalModelId,
            name: finalName,
            description: "Transferred model package",
            size: "Local",
            isDownloaded: true,
            localPath: importedPath.path,
            metaYamlPath: importedPath.appendingPathComponent("meta.yaml").path,
            sourceKind: .localImported
        )

        availableModels.removeAll { $0.id == newModel.id }
        availableModels.append(newModel)
        markModelAsJustAdded(newModel.id)
        try await StorageService.shared.saveModelsRegistry(availableModels)
        cleanupIncomingPackageArtifactsIfNeeded(incomingURL: incomingURL, resolvedPackageRoot: packageRoot)
        return newModel
    }

    private func resolveIncomingPackageRoot(from incomingURL: URL) throws -> URL {
        let fileManager = FileManager.default
        let standardized = incomingURL.standardizedFileURL

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: standardized.path, isDirectory: &isDirectory) else {
            throw ModelPackageImportError.invalidPackageRoot
        }

        if isDirectory.boolValue {
            let manifestAtRoot = standardized.appendingPathComponent("manifest.json")
            if fileManager.fileExists(atPath: manifestAtRoot.path) {
                return standardized
            }

            let children = (try? fileManager.contentsOfDirectory(at: standardized, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])) ?? []
            if let childWithManifest = children.first(where: {
                var childIsDir: ObjCBool = false
                guard fileManager.fileExists(atPath: $0.path, isDirectory: &childIsDir), childIsDir.boolValue else { return false }
                return fileManager.fileExists(atPath: $0.appendingPathComponent("manifest.json").path)
            }) {
                return childWithManifest
            }
        }

        throw ModelPackageImportError.invalidPackageRoot
    }

    private func validatePackageCompatibility(_ manifest: ModelPackageManifest) throws {
        guard manifest.formatVersion == 1 else {
            throw ModelPackageImportError.unsupportedFormatVersion(manifest.formatVersion)
        }

        if let minAppVersion = manifest.minAppVersion, !minAppVersion.isEmpty {
            let currentVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
            if compareVersionStrings(currentVersion, minAppVersion) == .orderedAscending {
                throw ModelPackageImportError.incompatibleAppVersion(minimum: minAppVersion, current: currentVersion)
            }
        }
    }

    private func validatePackageFiles(manifest: ModelPackageManifest, packageRoot: URL) async throws {
        try await Task.detached(priority: .userInitiated) {
            let fileManager = FileManager.default
            for entry in manifest.files {
                guard let normalizedEntryPath = Self.normalizePackageRelativePath(entry.path) else {
                    throw ModelPackageImportError.invalidManifest
                }
                guard let fileURL = Self.resolvePackageFileURL(
                    for: normalizedEntryPath,
                    packageRoot: packageRoot,
                    fileManager: fileManager
                ) else {
                    throw ModelPackageImportError.missingFile(path: entry.path)
                }

                if let expectedSize = entry.sizeBytes,
                   let attrs = try? fileManager.attributesOfItem(atPath: fileURL.path),
                   let size = attrs[.size] as? Int64,
                   size != expectedSize {
                    throw ModelPackageImportError.fileHashMismatch(path: entry.path)
                }

                let data = try Data(contentsOf: fileURL, options: [.mappedIfSafe])
                let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                if hash.lowercased() != entry.sha256.lowercased() {
                    throw ModelPackageImportError.fileHashMismatch(path: entry.path)
                }
            }
        }.value
    }

    private func compareVersionStrings(_ lhs: String, _ rhs: String) -> ComparisonResult {
        let lhsParts = lhs.split(separator: ".").map { Int($0) ?? 0 }
        let rhsParts = rhs.split(separator: ".").map { Int($0) ?? 0 }
        let maxCount = max(lhsParts.count, rhsParts.count)

        for i in 0..<maxCount {
            let l = i < lhsParts.count ? lhsParts[i] : 0
            let r = i < rhsParts.count ? rhsParts[i] : 0
            if l < r { return .orderedAscending }
            if l > r { return .orderedDescending }
        }
        return .orderedSame
    }

    private func shouldProcessIncomingTransferURL(_ url: URL, dedupeWindow: TimeInterval = 10) -> Bool {
        let now = Date()
        recentlyHandledIncomingURLKeys = recentlyHandledIncomingURLKeys.filter { now.timeIntervalSince($0.value) <= dedupeWindow }
        let key = url.standardizedFileURL.path
        if recentlyHandledIncomingURLKeys[key] != nil {
            return false
        }
        recentlyHandledIncomingURLKeys[key] = now
        return true
    }

    private func cleanupIncomingPackageArtifactsIfNeeded(incomingURL: URL, resolvedPackageRoot: URL) {
        let fileManager = FileManager.default
        let documentsRoot = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0].standardizedFileURL
        let modelsRoot = documentsRoot.appendingPathComponent("Models", isDirectory: true).standardizedFileURL
        let conversationsRoot = documentsRoot.appendingPathComponent("Conversations", isDirectory: true).standardizedFileURL
        let inboxRoot = documentsRoot
            .appendingPathComponent("Inbox", isDirectory: true)
            .standardizedFileURL
        let tempRoot = fileManager.temporaryDirectory.standardizedFileURL
        let cachesRoot = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0].standardizedFileURL

        let candidates = [incomingURL.standardizedFileURL, resolvedPackageRoot.standardizedFileURL]
        var seen = Set<String>()
        for candidate in candidates where seen.insert(candidate.path).inserted {
            guard shouldDeleteIncomingPackageArtifact(
                candidate,
                inboxRoot: inboxRoot,
                tempRoot: tempRoot,
                cachesRoot: cachesRoot,
                modelsRoot: modelsRoot,
                conversationsRoot: conversationsRoot
            ) else {
                continue
            }
            guard fileManager.fileExists(atPath: candidate.path) else { continue }
            if removeItemBestEffort(candidate, fileManager: fileManager) {
                logInfo("Cleaned up imported transfer artifact: \(candidate.lastPathComponent)", category: .model)
            }
        }
    }

    private func shouldDeleteIncomingPackageArtifact(
        _ url: URL,
        inboxRoot: URL,
        tempRoot: URL,
        cachesRoot: URL,
        modelsRoot: URL,
        conversationsRoot: URL
    ) -> Bool {
        let normalized = url.standardizedFileURL.path
        if isPath(normalized, inside: modelsRoot.path) || isPath(normalized, inside: conversationsRoot.path) {
            return false
        }

        if url.pathExtension.lowercased() == "anemllpkg" {
            return true
        }

        return isPath(normalized, inside: inboxRoot.path)
            || isPath(normalized, inside: tempRoot.path)
            || isPath(normalized, inside: cachesRoot.path)
    }

    private func removeItemBestEffort(_ url: URL, fileManager: FileManager) -> Bool {
        var coordinationError: NSError?
        var removed = false

        let coordinator = NSFileCoordinator()
        coordinator.coordinate(writingItemAt: url, options: .forDeleting, error: &coordinationError) { coordinatedURL in
            do {
                if fileManager.fileExists(atPath: coordinatedURL.path) {
                    try fileManager.removeItem(at: coordinatedURL)
                }
                removed = true
            } catch {
                logWarning("Failed coordinated cleanup for \(coordinatedURL.path): \(error)", category: .model)
            }
        }

        if removed {
            return true
        }
        if let coordinationError {
            logWarning("File coordination failed for cleanup \(url.path): \(coordinationError)", category: .model)
        }

        do {
            if fileManager.fileExists(atPath: url.path) {
                try fileManager.removeItem(at: url)
            }
            return true
        } catch {
            logWarning("Failed to clean up transfer artifact \(url.path): \(error)", category: .model)
            return false
        }
    }

    private func isPath(_ child: String, inside parent: String) -> Bool {
        if child == parent { return true }
        return child.hasPrefix(parent.hasSuffix("/") ? parent : parent + "/")
    }
    #endif

    // MARK: - Package Share (macOS sender)

    #if os(macOS)
    func shareModelToIOS(_ model: ModelInfo) async {
        do {
            guard model.isDownloaded, let localPath = model.localPath else {
                throw LocalModelValidationError.invalidStructure(["Model files are unavailable."])
            }

            let sourceRoot = URL(fileURLWithPath: localPath, isDirectory: true)
            let packageURL = try await buildTransferPackage(for: model, sourceRoot: sourceRoot)
            preparedTransferPackages.append(packageURL)
            scheduleTransferPackageCleanup(packageURL)

            if let airDrop = NSSharingService(named: .sendViaAirDrop) {
                airDrop.perform(withItems: [packageURL])
                logInfo("Started AirDrop share for model package: \(packageURL.lastPathComponent)", category: .model)
            } else {
                NSWorkspace.shared.activateFileViewerSelecting([packageURL])
                errorMessage = "AirDrop service unavailable. Opened package in Finder."
            }
        } catch {
            errorMessage = error.localizedDescription
            logError("Failed to share model package: \(error)", category: .model)
        }
    }

    private func buildTransferPackage(for model: ModelInfo, sourceRoot: URL) async throws -> URL {
        let validatedRoot = try detectLocalModelRoot(from: sourceRoot)
        let fileManager = FileManager.default
        cleanupStaleTransferPackages()
        let now = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let safeModelName = slugify(model.name)

        let packageRoot = fileManager.temporaryDirectory
            .appendingPathComponent("anemll-transfer", isDirectory: true)
            .appendingPathComponent("\(safeModelName)-\(now).anemllpkg", isDirectory: true)
        let modelOut = packageRoot.appendingPathComponent("model", isDirectory: true)

        if fileManager.fileExists(atPath: packageRoot.path) {
            try fileManager.removeItem(at: packageRoot)
        }

        try fileManager.createDirectory(at: packageRoot, withIntermediateDirectories: true)
        try fileManager.copyItem(at: validatedRoot, to: modelOut)

        let fileEntries = try buildManifestEntries(forModelRoot: modelOut, packageRootPathPrefix: "model")
        let appVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
        let manifest = ModelPackageManifest(
            formatVersion: 1,
            modelName: model.name,
            modelId: model.id,
            modelRootPath: "model",
            minAppVersion: appVersion,
            files: fileEntries
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let manifestData = try encoder.encode(manifest)
        try manifestData.write(to: packageRoot.appendingPathComponent("manifest.json"), options: .atomic)

        return packageRoot
    }

    private func buildManifestEntries(forModelRoot modelRoot: URL, packageRootPathPrefix: String) throws -> [ModelPackageFileEntry] {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: modelRoot,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        guard let normalizedPrefix = Self.normalizePackageRelativePath(packageRootPathPrefix) else {
            return []
        }

        var entries: [ModelPackageFileEntry] = []
        for case let fileURL as URL in enumerator {
            let values = try fileURL.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
            if values.isDirectory == true {
                continue
            }

            guard let relative = relativePath(from: modelRoot, to: fileURL),
                  let normalizedRelative = Self.normalizePackageRelativePath(relative) else {
                logWarning("Skipping file with invalid relative path in transfer package: \(fileURL.path)", category: .model)
                continue
            }

            let manifestPath = "\(normalizedPrefix)/\(normalizedRelative)"
            let sha = try sha256Hex(for: fileURL)
            entries.append(ModelPackageFileEntry(path: manifestPath, sha256: sha, sizeBytes: values.fileSize.map(Int64.init)))
        }

        return entries.sorted { $0.path < $1.path }
    }

    private func cleanupStaleTransferPackages(maxAge: TimeInterval = 24 * 60 * 60) {
        let fileManager = FileManager.default
        let transferRoot = fileManager.temporaryDirectory.appendingPathComponent("anemll-transfer", isDirectory: true)
        guard let entries = try? fileManager.contentsOfDirectory(
            at: transferRoot,
            includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            preparedTransferPackages.removeAll { !fileManager.fileExists(atPath: $0.path) }
            return
        }

        let cutoff = Date().addingTimeInterval(-maxAge)
        for entry in entries where entry.pathExtension.lowercased() == "anemllpkg" {
            let values = try? entry.resourceValues(forKeys: [.isDirectoryKey, .contentModificationDateKey])
            guard values?.isDirectory == true else { continue }
            if let modified = values?.contentModificationDate, modified < cutoff {
                try? fileManager.removeItem(at: entry)
            }
        }

        preparedTransferPackages.removeAll { !fileManager.fileExists(atPath: $0.path) }
    }

    private func scheduleTransferPackageCleanup(_ packageURL: URL, delaySeconds: TimeInterval = 45 * 60) {
        let packagePath = packageURL.standardizedFileURL.path
        Task { @MainActor [weak self] in
            let delay = max(delaySeconds, 0)
            let nanoseconds = UInt64(delay * 1_000_000_000)
            try? await Task.sleep(nanoseconds: nanoseconds)

            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: packagePath) {
                try? fileManager.removeItem(atPath: packagePath)
            }

            self?.preparedTransferPackages.removeAll {
                let trackedPath = $0.standardizedFileURL.path
                return trackedPath == packagePath || !fileManager.fileExists(atPath: trackedPath)
            }
        }
    }

    private func relativePath(from root: URL, to file: URL) -> String? {
        let rootPath = root.resolvingSymlinksInPath().standardizedFileURL.path
        let filePath = file.resolvingSymlinksInPath().standardizedFileURL.path

        if filePath == rootPath {
            return ""
        }
        if filePath.hasPrefix(rootPath + "/") {
            return String(filePath.dropFirst(rootPath.count + 1))
        }

        // Fallback for path representation mismatches.
        let rootComponents = root.standardizedFileURL.pathComponents
        let fileComponents = file.standardizedFileURL.pathComponents
        guard fileComponents.count >= rootComponents.count else {
            return nil
        }
        guard Array(fileComponents.prefix(rootComponents.count)) == rootComponents else {
            return nil
        }

        return fileComponents.dropFirst(rootComponents.count).joined(separator: "/")
    }

    private func sha256Hex(for fileURL: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer {
            try? handle.close()
        }

        var hasher = SHA256()
        while true {
            let data = try handle.read(upToCount: 1_048_576) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }
    #endif

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

    nonisolated private static func normalizePackageRelativePath(_ rawPath: String) -> String? {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let unified = trimmed.replacingOccurrences(of: "\\", with: "/")
        let components = unified.split(separator: "/", omittingEmptySubsequences: true)
        if components.isEmpty {
            return nil
        }

        var cleaned: [String] = []
        cleaned.reserveCapacity(components.count)

        for raw in components {
            let part = String(raw)
            if part == "." {
                continue
            }
            if part == ".." {
                return nil
            }
            cleaned.append(part)
        }

        guard !cleaned.isEmpty else { return nil }
        return cleaned.joined(separator: "/")
    }

    nonisolated private static func resolvePackageFileURL(
        for normalizedPath: String,
        packageRoot: URL,
        fileManager: FileManager
    ) -> URL? {
        var candidates: [String] = [normalizedPath]
        candidates.append(contentsOf: legacyPathCandidates(for: normalizedPath))

        var seen = Set<String>()
        for candidate in candidates where seen.insert(candidate).inserted {
            let url = packageRoot.appendingPathComponent(candidate)
            if fileManager.fileExists(atPath: url.path) {
                return url
            }
        }
        return nil
    }

    nonisolated private static func legacyPathCandidates(for normalizedPath: String) -> [String] {
        let components = normalizedPath.split(separator: "/").map(String.init)
        guard components.count >= 2 else { return [] }

        let root = components[0]
        var candidates: [String] = []

        // Compatibility for older sender bug that produced model//private*.*
        if components[1].hasPrefix("private"), components[1].count > "private".count {
            var fixed = components
            fixed[1] = String(fixed[1].dropFirst("private".count))
            if !fixed[1].isEmpty {
                candidates.append(fixed.joined(separator: "/"))
            }
        }

        // Compatibility when an absolute path fragment leaked into manifest and contains /model/... twice.
        let marker = "/\(root)/"
        if let range = normalizedPath.range(of: marker, options: .backwards) {
            let suffix = String(normalizedPath[range.upperBound...])
            if !suffix.isEmpty {
                candidates.append("\(root)/\(suffix)")
            }
        }

        return candidates
    }

    private func refreshedModelStatus(for inputModel: ModelInfo) async -> ModelInfo {
        var model = inputModel
        // Reset transient download state (downloads don't survive app restart)
        model.isDownloading = false
        model.downloadProgress = nil

        switch model.sourceKind {
        case .huggingFace, .localImported:
            let isDownloaded = await StorageService.shared.isModelDownloaded(model.id)
            model.isDownloaded = isDownloaded

            if isDownloaded {
                let path = await StorageService.shared.modelPath(for: model.id)
                model.localPath = path.path
                model.metaYamlPath = path.appendingPathComponent("meta.yaml").path
                model.downloadError = nil
                logDebug("Model \(model.name) is available at \(path.path)", category: .model)
            } else if model.sourceKind == .localImported {
                model.downloadError = "Imported model files are missing."
            } else {
                model.downloadError = nil
            }

        case .localLinked:
            let resolvedPath = resolveLinkedModelPath(for: model) ?? model.linkedPath ?? model.localPath
            guard let resolvedPath else {
                model.isDownloaded = false
                model.downloadError = "Linked source folder is not configured."
                return model
            }

            let linkedRoot = URL(fileURLWithPath: resolvedPath)
            let issues = validationIssues(forModelRoot: linkedRoot)
            model.localPath = linkedRoot.path
            model.linkedPath = linkedRoot.path
            model.metaYamlPath = linkedRoot.appendingPathComponent("meta.yaml").path
            model.isDownloaded = issues.isEmpty
            model.downloadError = issues.isEmpty ? nil : "Linked source folder missing: \(issues.joined(separator: ", "))"
        }

        return model
    }

    private func detectLocalModelRoot(from droppedURL: URL) throws -> URL {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: droppedURL.path, isDirectory: &isDirectory) else {
            throw LocalModelValidationError.invalidStructure([
                "Folder does not exist at path \(droppedURL.path)."
            ])
        }
        guard isDirectory.boolValue else {
            throw LocalModelValidationError.notDirectory
        }

        let startURL = droppedURL.standardizedFileURL
        var candidates: [URL] = []
        candidates.append(startURL)
        candidates.append(startURL.appendingPathComponent("ios", isDirectory: true))
        candidates.append(startURL.appendingPathComponent("hf", isDirectory: true).appendingPathComponent("ios", isDirectory: true))

        // Check parents too in case user drops a containing folder above the model root.
        var parent = startURL.deletingLastPathComponent()
        for _ in 0..<3 {
            candidates.append(parent)
            candidates.append(parent.appendingPathComponent("hf", isDirectory: true).appendingPathComponent("ios", isDirectory: true))
            let nextParent = parent.deletingLastPathComponent()
            if nextParent == parent { break }
            parent = nextParent
        }

        for candidate in deduplicatedURLs(candidates) {
            if validationIssues(forModelRoot: candidate).isEmpty {
                return candidate
            }
        }

        if let descendant = findDescendantModelRoot(from: startURL, maxDepth: 3) {
            return descendant
        }

        throw LocalModelValidationError.invalidStructure([
            "Expected a model folder containing meta.yaml and at least one .mlmodelc directory.",
            "Dropped path: \(droppedURL.path)"
        ])
    }

    private func findDescendantModelRoot(from startURL: URL, maxDepth: Int) -> URL? {
        let fileManager = FileManager.default
        var queue: [(url: URL, depth: Int)] = [(startURL, 0)]
        var visited = Set<String>()

        while !queue.isEmpty {
            let (currentURL, depth) = queue.removeFirst()
            if visited.contains(currentURL.path) { continue }
            visited.insert(currentURL.path)

            if validationIssues(forModelRoot: currentURL).isEmpty {
                return currentURL
            }

            guard depth < maxDepth else { continue }
            guard let children = try? fileManager.contentsOfDirectory(
                at: currentURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }

            for child in children {
                if let values = try? child.resourceValues(forKeys: [.isDirectoryKey]), values.isDirectory == true {
                    queue.append((child, depth + 1))
                }
            }
        }

        return nil
    }

    private func validationIssues(forModelRoot url: URL) -> [String] {
        let fileManager = FileManager.default
        var issues: [String] = []

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return ["not a folder"]
        }

        let metaYaml = url.appendingPathComponent("meta.yaml")
        if !fileManager.fileExists(atPath: metaYaml.path) {
            issues.append("meta.yaml")
        }

        let hasMLModelc: Bool = ((try? fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? [])
            .contains(where: { $0.pathExtension.lowercased() == "mlmodelc" })
        if !hasMLModelc {
            issues.append("*.mlmodelc")
        }

        return issues
    }

    private func suggestModelName(from droppedURL: URL) -> String {
        let components = droppedURL.standardizedFileURL.pathComponents.filter { $0 != "/" && !$0.isEmpty }
        if components.count >= 3 {
            let leaf = components[components.count - 1]
            let parent = components[components.count - 2]

            // Prefer grandparent for common converter/export layouts like:
            // .../<name>/hf/ios and .../<name>/hf_dist/ios
            if isIOSFolderName(leaf) && isHFExportFolderName(parent) {
                return beautifyDisplayName(components[components.count - 3])
            }
        }

        for component in components.reversed() {
            if !isGenericModelFolderName(component) {
                return beautifyDisplayName(component)
            }
        }

        return "Model"
    }

    private func beautifyDisplayName(_ raw: String) -> String {
        let cleaned = raw
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty ? "Model" : cleaned
    }

    private func isIOSFolderName(_ value: String) -> Bool {
        normalizedFolderToken(value) == "ios"
    }

    private func isHFExportFolderName(_ value: String) -> Bool {
        let lower = value.lowercased()
        let normalized = normalizedFolderToken(value)

        if lower == "hf" || lower == "huggingface" {
            return true
        }
        if lower.hasPrefix("hf_") || lower.hasPrefix("hf-") {
            return true
        }
        if normalized.hasPrefix("hf") && normalized.hasSuffix("dist") {
            return true
        }
        return false
    }

    private func isGenericModelFolderName(_ value: String) -> Bool {
        let lower = value.lowercased()
        let normalized = normalizedFolderToken(value)
        let genericExact: Set<String> = [
            "ios", "hf", "huggingface",
            "model", "models",
            "output", "outputs",
            "converted", "convert",
            "dist", "build"
        ]

        if genericExact.contains(lower) || genericExact.contains(normalized) {
            return true
        }
        if isHFExportFolderName(lower) {
            return true
        }
        return false
    }

    private func normalizedFolderToken(_ value: String) -> String {
        value
            .lowercased()
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: " ", with: "")
    }

    private func uniqueDisplayName(for requestedName: String) -> String {
        let base = beautifyDisplayName(requestedName)
        let existing = Set(availableModels.map { $0.name.lowercased() })
        if !existing.contains(base.lowercased()) {
            return base
        }

        var suffix = 2
        while true {
            let candidate = "\(base)-\(suffix)"
            if !existing.contains(candidate.lowercased()) {
                return candidate
            }
            suffix += 1
        }
    }

    private func uniqueLocalModelId(forDisplayName displayName: String) -> String {
        let baseSlug = slugify(displayName)
        var candidate = "local/\(baseSlug)"
        let existingIds = Set(availableModels.map { $0.id })
        if !existingIds.contains(candidate) {
            return candidate
        }

        var suffix = 2
        while true {
            candidate = "local/\(baseSlug)-\(suffix)"
            if !existingIds.contains(candidate) {
                return candidate
            }
            suffix += 1
        }
    }

    private func slugify(_ value: String) -> String {
        let lower = value.lowercased()
        let allowed = CharacterSet.alphanumerics
        var buffer = ""
        var previousWasHyphen = false

        for scalar in lower.unicodeScalars {
            if allowed.contains(scalar) {
                buffer.append(Character(scalar))
                previousWasHyphen = false
            } else if !previousWasHyphen {
                buffer.append("-")
                previousWasHyphen = true
            }
        }

        let trimmed = buffer.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return trimmed.isEmpty ? "model" : trimmed
    }

    private func deduplicatedURLs(_ urls: [URL]) -> [URL] {
        var seen = Set<String>()
        var result: [URL] = []
        for url in urls {
            if seen.insert(url.path).inserted {
                result.append(url)
            }
        }
        return result
    }

    private func resolveLinkedModelPath(for model: ModelInfo) -> String? {
        #if os(macOS)
        if let base64 = model.bookmarkDataBase64, let bookmarkData = Data(base64Encoded: base64) {
            var isStale = false
            if let resolvedURL = try? URL(
                resolvingBookmarkData: bookmarkData,
                options: [.withSecurityScope, .withoutUI],
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            ) {
                return resolvedURL.path
            }
        }
        #endif
        return model.linkedPath ?? model.localPath
    }

    private func markModelAsJustAdded(_ id: String) {
        justAddedModelId = id
        clearJustAddedTask?.cancel()
        clearJustAddedTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(90))
            guard let self else { return }
            if self.justAddedModelId == id {
                self.justAddedModelId = nil
            }
        }
    }

    private func makeBookmarkDataBase64(for url: URL) -> String? {
        #if os(macOS)
        do {
            let bookmark = try url.bookmarkData(options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil)
            return bookmark.base64EncodedString()
        } catch {
            logWarning("Failed to create security-scoped bookmark: \(error)", category: .model)
            return nil
        }
        #else
        return nil
        #endif
    }

    private func updateModelDownloading(_ id: String, isDownloading: Bool) {
        if let index = availableModels.firstIndex(where: { $0.id == id }) {
            availableModels[index].isDownloading = isDownloading
            if !isDownloading {
                availableModels[index].downloadProgress = nil
                if availableModels[index].sourceKind == .huggingFace {
                    availableModels[index].downloadError = nil
                }
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
