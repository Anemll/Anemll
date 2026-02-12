//
//  StorageService.swift
//  ANEMLLChat
//
//  Persistence for conversations and settings
//

import Foundation
#if os(iOS)
import UIKit
#endif

#if os(macOS)
enum StorageMigrationMode: String, Sendable {
    case copy
    case move
}

struct StorageMigrationProgress: Sendable {
    let completedUnits: Int
    let totalUnits: Int
    let message: String

    var fractionCompleted: Double {
        guard totalUnits > 0 else { return 0 }
        return Double(completedUnits) / Double(totalUnits)
    }
}
#endif

/// Errors that can occur during storage operations
enum StorageError: LocalizedError {
    case encodingFailed
    case decodingFailed
    case fileWriteFailed(Error)
    case fileReadFailed(Error)
    case directoryCreationFailed(Error)

    var errorDescription: String? {
        switch self {
        case .encodingFailed: return "Failed to encode data"
        case .decodingFailed: return "Failed to decode data"
        case .fileWriteFailed(let error): return "Failed to write file: \(error.localizedDescription)"
        case .fileReadFailed(let error): return "Failed to read file: \(error.localizedDescription)"
        case .directoryCreationFailed(let error): return "Failed to create directory: \(error.localizedDescription)"
        }
    }
}

/// Service for persisting app data
actor StorageService {
    static let shared = StorageService()

    private let fileManager = FileManager.default
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    #if os(macOS)
    private static let macOSStorageFolderPathKey = "macOSStorageFolderPath"
    private static let macOSLegacyMigrationStatePrefix = "macOSLegacyStorageMigratedTo"
    #endif

    private init() {
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        decoder.dateDecodingStrategy = .iso8601

        #if os(macOS)
        Task { [weak self] in
            await self?.migrateLegacyMacOSStorageIfNeeded()
        }
        #endif
    }

    // MARK: - Directories

    /// Documents directory URL (sandboxed on iOS, user's Documents on macOS)
    private var documentsDirectory: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// App data root directory
    /// - macOS: ~/.cache/anemll
    /// - iOS: app Documents directory
    private var appDataRootDirectory: URL {
        #if os(macOS)
        return configuredMacOSStorageRootDirectory
        #else
        return documentsDirectory
        #endif
    }

    #if os(macOS)
    private var defaultMacOSStorageRootDirectory: URL {
        fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache", isDirectory: true)
            .appendingPathComponent("anemll", isDirectory: true)
            .standardizedFileURL
    }

    private var configuredMacOSStorageRootDirectory: URL {
        if let storedPath = UserDefaults.standard.string(forKey: Self.macOSStorageFolderPathKey),
           !storedPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: storedPath, isDirectory: true).standardizedFileURL
        }
        return defaultMacOSStorageRootDirectory
    }
    #endif

    /// Conversations directory
    private var conversationsDirectory: URL {
        appDataRootDirectory.appendingPathComponent("Conversations", isDirectory: true)
    }

    /// Models directory (for downloaded models)
    /// - macOS: ~/.cache/anemll/Models
    /// - iOS: Documents/Models/ (in app's sandboxed Documents)
    var modelsDirectory: URL {
        #if os(macOS)
        return appDataRootDirectory.appendingPathComponent("Models", isDirectory: true)
        #else
        // iOS: Store in app's Documents/Models (visible in Files app with UIFileSharingEnabled)
        return documentsDirectory.appendingPathComponent("Models", isDirectory: true)
        #endif
    }

    /// Ensure directory exists
    private func ensureDirectoryExists(_ url: URL) throws {
        if !fileManager.fileExists(atPath: url.path) {
            do {
                try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
                logDebug("Created directory: \(url.path)", category: .storage)
            } catch {
                throw StorageError.directoryCreationFailed(error)
            }
        }
    }

    #if os(macOS)
    func currentMacOSStorageFolderURL() -> URL {
        configuredMacOSStorageRootDirectory
    }

    /// Set macOS storage root folder.
    /// Returns old/new URLs and whether the folder actually changed.
    @discardableResult
    func updateMacOSStorageFolder(to newFolderURL: URL) throws -> (oldURL: URL, newURL: URL, changed: Bool) {
        let oldURL = configuredMacOSStorageRootDirectory
        let normalizedNewURL = newFolderURL.standardizedFileURL

        try ensureDirectoryExists(normalizedNewURL)

        if normalizedNewURL.path == oldURL.path {
            return (oldURL, normalizedNewURL, false)
        }

        UserDefaults.standard.set(normalizedNewURL.path, forKey: Self.macOSStorageFolderPathKey)
        logInfo("Updated macOS storage folder to \(normalizedNewURL.path)", category: .storage)
        return (oldURL, normalizedNewURL, true)
    }

    /// Migrate app data between macOS storage roots.
    /// `copy` leaves original data; `move` removes source items after transfer.
    func migrateMacOSStorage(
        from oldRoot: URL,
        to newRoot: URL,
        mode: StorageMigrationMode,
        progress: (@Sendable (StorageMigrationProgress) -> Void)? = nil
    ) throws {
        let source = oldRoot.standardizedFileURL
        let destination = newRoot.standardizedFileURL
        guard source.path != destination.path else { return }

        let sourceModelsRoot = source.appendingPathComponent("Models", isDirectory: true)
        let modelDirectories = try listTopLevelDirectories(at: sourceModelsRoot)
        let modelWorkUnits = max(modelDirectories.count, 1)
        let totalUnits = 2 + modelWorkUnits
        var completedUnits = 0

        emitMigrationProgress(progress, completedUnits: completedUnits, totalUnits: totalUnits, message: "Preparing migration...")

        try ensureDirectoryExists(destination)

        try migrateDirectoryContents(
            source: source.appendingPathComponent("Conversations", isDirectory: true),
            destination: destination.appendingPathComponent("Conversations", isDirectory: true),
            mode: mode,
            protectedRoot: source
        )
        completedUnits += 1
        emitMigrationProgress(progress, completedUnits: completedUnits, totalUnits: totalUnits, message: "Migrated conversations")

        try migrateModelsRegistryFile(
            source: source.appendingPathComponent("models.json"),
            destination: destination.appendingPathComponent("models.json"),
            mode: mode
        )
        completedUnits += 1
        emitMigrationProgress(progress, completedUnits: completedUnits, totalUnits: totalUnits, message: "Migrated model registry")

        let destinationModelsRoot = destination.appendingPathComponent("Models", isDirectory: true)
        try ensureDirectoryExists(destinationModelsRoot)

        if modelDirectories.isEmpty {
            completedUnits += 1
            emitMigrationProgress(progress, completedUnits: completedUnits, totalUnits: totalUnits, message: "No model folders to migrate")
        } else {
            for (index, sourceModelDir) in modelDirectories.enumerated() {
                let destinationModelDir = destinationModelsRoot.appendingPathComponent(sourceModelDir.lastPathComponent, isDirectory: true)
                try migrateDirectoryContents(source: sourceModelDir, destination: destinationModelDir, mode: mode, protectedRoot: source)
                completedUnits += 1
                emitMigrationProgress(
                    progress,
                    completedUnits: completedUnits,
                    totalUnits: totalUnits,
                    message: "Migrated model folder \(index + 1)/\(modelDirectories.count): \(sourceModelDir.lastPathComponent)"
                )
            }
        }
    }

    private func migrateLegacyMacOSStorageIfNeeded() {
        let targetRoot = configuredMacOSStorageRootDirectory
        let migrationStateKey = "\(Self.macOSLegacyMigrationStatePrefix)::\(targetRoot.path)"
        if UserDefaults.standard.bool(forKey: migrationStateKey) {
            return
        }

        let legacyDocuments = fileManager.homeDirectoryForCurrentUser.appendingPathComponent("Documents", isDirectory: true)

        do {
            try ensureDirectoryExists(targetRoot)

            // Legacy chats and registry lived directly under ~/Documents.
            try migrateDirectoryContents(
                source: legacyDocuments.appendingPathComponent("Conversations", isDirectory: true),
                destination: targetRoot.appendingPathComponent("Conversations", isDirectory: true),
                mode: .copy
            )
            try migrateModelsRegistryFile(
                source: legacyDocuments.appendingPathComponent("models.json"),
                destination: targetRoot.appendingPathComponent("models.json"),
                mode: .copy
            )

            // Legacy downloaded models were top-level folders in ~/Documents.
            try ensureDirectoryExists(targetRoot.appendingPathComponent("Models", isDirectory: true))
            if let entries = try? fileManager.contentsOfDirectory(
                at: legacyDocuments,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) {
                for entry in entries {
                    guard isLegacyModelDirectory(entry) else { continue }
                    let destination = targetRoot
                        .appendingPathComponent("Models", isDirectory: true)
                        .appendingPathComponent(entry.lastPathComponent, isDirectory: true)
                    try migrateDirectoryContents(source: entry, destination: destination, mode: .copy)
                }
            }

            UserDefaults.standard.set(true, forKey: migrationStateKey)
            logInfo("Completed macOS legacy storage migration to \(targetRoot.path)", category: .storage)
        } catch {
            logWarning("Legacy macOS storage migration failed: \(error)", category: .storage)
        }
    }

    private func isLegacyModelDirectory(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return false
        }

        let hasMeta = fileManager.fileExists(atPath: url.appendingPathComponent("meta.yaml").path)
        guard hasMeta else { return false }

        let children = (try? fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? []
        let hasCompiledModel = children.contains { $0.pathExtension.lowercased() == "mlmodelc" }
        return hasCompiledModel
    }

    private func migrateDirectoryContents(
        source: URL,
        destination: URL,
        mode: StorageMigrationMode,
        protectedRoot: URL? = nil
    ) throws {
        var sourceIsDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: source.path, isDirectory: &sourceIsDirectory), sourceIsDirectory.boolValue else {
            return
        }

        if !fileManager.fileExists(atPath: destination.path) {
            try ensureDirectoryExists(destination.deletingLastPathComponent())
            switch mode {
            case .copy:
                try fileManager.copyItem(at: source, to: destination)
            case .move:
                try fileManager.moveItem(at: source, to: destination)
            }
            return
        }

        try ensureDirectoryExists(destination)
        let sourceChildren = try fileManager.contentsOfDirectory(
            at: source,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        for child in sourceChildren {
            let target = destination.appendingPathComponent(child.lastPathComponent, isDirectory: true)
            var childIsDirectory: ObjCBool = false
            let childExists = fileManager.fileExists(atPath: child.path, isDirectory: &childIsDirectory)
            guard childExists else { continue }

            var targetIsDirectory: ObjCBool = false
            let targetExists = fileManager.fileExists(atPath: target.path, isDirectory: &targetIsDirectory)

            if targetExists {
                if childIsDirectory.boolValue && targetIsDirectory.boolValue {
                    try migrateDirectoryContents(source: child, destination: target, mode: mode, protectedRoot: protectedRoot)
                    if mode == .move {
                        try? fileManager.removeItem(at: child)
                    }
                } else {
                    // Destination takes precedence; on move, drop duplicate source entry.
                    if mode == .move {
                        try? fileManager.removeItem(at: child)
                    }
                }
                continue
            }

            switch mode {
            case .copy:
                try fileManager.copyItem(at: child, to: target)
            case .move:
                try fileManager.moveItem(at: child, to: target)
            }
        }

        if mode == .move {
            let normalizedSource = source.standardizedFileURL.path
            let normalizedProtected = protectedRoot?.standardizedFileURL.path
            if normalizedSource == normalizedProtected {
                // Safety guard: never remove the selected source root folder itself.
                return
            }
            try? fileManager.removeItem(at: source)
        }
    }

    private func migrateModelsRegistryFile(source: URL, destination: URL, mode: StorageMigrationMode) throws {
        guard fileManager.fileExists(atPath: source.path) else { return }

        if !fileManager.fileExists(atPath: destination.path) {
            try ensureDirectoryExists(destination.deletingLastPathComponent())
            switch mode {
            case .copy:
                try fileManager.copyItem(at: source, to: destination)
            case .move:
                try fileManager.moveItem(at: source, to: destination)
            }
            return
        }

        // Merge by model id when both registry files exist.
        let sourceData = try Data(contentsOf: source)
        let destinationData = try Data(contentsOf: destination)
        let sourceModels = (try? decoder.decode([ModelInfo].self, from: sourceData)) ?? []
        var destinationModels = (try? decoder.decode([ModelInfo].self, from: destinationData)) ?? []

        for sourceModel in sourceModels where !destinationModels.contains(where: { $0.id == sourceModel.id }) {
            destinationModels.append(sourceModel)
        }

        let merged = try encoder.encode(destinationModels)
        try merged.write(to: destination, options: .atomic)

        if mode == .move {
            try? fileManager.removeItem(at: source)
        }
    }

    private func emitMigrationProgress(
        _ callback: (@Sendable (StorageMigrationProgress) -> Void)?,
        completedUnits: Int,
        totalUnits: Int,
        message: String
    ) {
        callback?(StorageMigrationProgress(completedUnits: completedUnits, totalUnits: totalUnits, message: message))
    }

    private func listTopLevelDirectories(at root: URL) throws -> [URL] {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return []
        }

        let entries = try fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        return entries
            .filter {
                (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            }
            .sorted { $0.lastPathComponent.localizedCaseInsensitiveCompare($1.lastPathComponent) == .orderedAscending }
    }
    #endif

    // MARK: - Conversations

    /// Save a conversation
    func saveConversation(_ conversation: Conversation) async throws {
        try ensureDirectoryExists(conversationsDirectory)

        let fileURL = conversationsDirectory.appendingPathComponent("\(conversation.id.uuidString).json")

        do {
            let data = try encoder.encode(conversation)
            try data.write(to: fileURL, options: .atomic)
            logDebug("Saved conversation: \(conversation.id)", category: .storage)
        } catch let error as EncodingError {
            logError("Encoding failed: \(error)", category: .storage)
            throw StorageError.encodingFailed
        } catch {
            logError("Write failed: \(error)", category: .storage)
            throw StorageError.fileWriteFailed(error)
        }
    }

    /// Load all conversations
    func loadConversations() async throws -> [Conversation] {
        try ensureDirectoryExists(conversationsDirectory)

        var conversations: [Conversation] = []

        do {
            let files = try fileManager.contentsOfDirectory(
                at: conversationsDirectory,
                includingPropertiesForKeys: nil
            )

            for file in files where file.pathExtension == "json" {
                do {
                    let data = try Data(contentsOf: file)
                    let conversation = try decoder.decode(Conversation.self, from: data)
                    conversations.append(conversation)
                } catch {
                    logWarning("Failed to load conversation \(file.lastPathComponent): \(error)", category: .storage)
                }
            }
        } catch {
            throw StorageError.fileReadFailed(error)
        }

        // Sort by most recent first
        conversations.sort { $0.updatedAt > $1.updatedAt }
        logInfo("Loaded \(conversations.count) conversations", category: .storage)

        return conversations
    }

    /// Delete a conversation
    func deleteConversation(_ id: UUID) async throws {
        let fileURL = conversationsDirectory.appendingPathComponent("\(id.uuidString).json")

        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                try fileManager.removeItem(at: fileURL)
                logDebug("Deleted conversation: \(id)", category: .storage)
            } catch {
                throw StorageError.fileWriteFailed(error)
            }
        }
    }

    // MARK: - Model Registry

    /// File for custom model registry
    private var modelsRegistryFile: URL {
        appDataRootDirectory.appendingPathComponent("models.json")
    }

    /// Save custom models to registry
    /// Only persists models that are locally imported/linked or actually downloaded.
    /// Non-downloaded HuggingFace models are not saved — the collection fetch is the
    /// authoritative source for those and saving them would create stale registry entries.
    func saveModelsRegistry(_ models: [ModelInfo]) async throws {
        try ensureDirectoryExists(appDataRootDirectory)

        // Only save: (a) not a hardcoded default, AND (b) locally imported/linked or downloaded
        let customModels = models.filter { model in
            guard !ModelInfo.defaultModels.contains(where: { $0.id == model.id }) else { return false }
            return model.sourceKind == .localImported
                || model.sourceKind == .localLinked
                || model.isDownloaded
        }

        do {
            let data = try encoder.encode(customModels)
            try data.write(to: modelsRegistryFile, options: .atomic)
            logDebug("Saved \(customModels.count) custom models to registry", category: .storage)
        } catch {
            throw StorageError.fileWriteFailed(error)
        }
    }

    /// Load custom models from registry
    func loadModelsRegistry() async throws -> [ModelInfo] {
        guard fileManager.fileExists(atPath: modelsRegistryFile.path) else {
            return []
        }

        do {
            let data = try Data(contentsOf: modelsRegistryFile)
            let models = try decoder.decode([ModelInfo].self, from: data)
            logInfo("Loaded \(models.count) custom models from registry", category: .storage)
            return models
        } catch {
            logWarning("Failed to load models registry: \(error)", category: .storage)
            return []
        }
    }

    // MARK: - Collection Cache

    /// File for cached HuggingFace collection models
    private var collectionCacheFile: URL {
        appDataRootDirectory.appendingPathComponent("collection_cache.json")
    }

    /// Save fetched collection models to cache
    func saveCollectionCache(_ models: [ModelInfo]) {
        do {
            try ensureDirectoryExists(appDataRootDirectory)
            let data = try encoder.encode(models)
            try data.write(to: collectionCacheFile, options: .atomic)
            logDebug("Saved \(models.count) collection models to cache", category: .storage)
        } catch {
            logWarning("Failed to save collection cache: \(error)", category: .storage)
        }
    }

    /// Load cached collection models (returns nil if no cache or decode error)
    func loadCollectionCache() -> [ModelInfo]? {
        guard fileManager.fileExists(atPath: collectionCacheFile.path) else {
            return nil
        }

        do {
            let data = try Data(contentsOf: collectionCacheFile)
            let models = try decoder.decode([ModelInfo].self, from: data)
            logInfo("Loaded \(models.count) models from collection cache", category: .storage)
            return models
        } catch {
            logWarning("Failed to load collection cache: \(error)", category: .storage)
            return nil
        }
    }

    // MARK: - Model Files

    /// Get local path for a model
    func modelPath(for modelId: String) -> URL {
        // Trim whitespace to prevent path issues from malformed model IDs
        let cleanId = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        return modelsDirectory.appendingPathComponent(cleanId.replacingOccurrences(of: "/", with: "_"))
    }

    /// Check if a model is downloaded
    func isModelDownloaded(_ modelId: String) async -> Bool {
        let modelDir = modelPath(for: modelId)
        let metaYaml = modelDir.appendingPathComponent("meta.yaml")
        return fileManager.fileExists(atPath: metaYaml.path)
    }

    /// Import a local model directory into app-managed storage using staged atomic finalize.
    /// This guarantees we never leave a partially imported model at the final destination.
    func importModelDirectory(from sourceURL: URL, toModelId modelId: String) async throws -> URL {
        try ensureDirectoryExists(modelsDirectory)

        let destinationURL = modelPath(for: modelId)
        let stagingURL = modelsDirectory.appendingPathComponent(".import-\(UUID().uuidString)", isDirectory: true)

        do {
            if fileManager.fileExists(atPath: stagingURL.path) {
                try fileManager.removeItem(at: stagingURL)
            }
            if fileManager.fileExists(atPath: destinationURL.path) {
                try fileManager.removeItem(at: destinationURL)
            }

            try fileManager.copyItem(at: sourceURL, to: stagingURL)
            try fileManager.moveItem(at: stagingURL, to: destinationURL)
            logInfo("Imported local model into app storage: \(destinationURL.path)", category: .storage)
            return destinationURL

        } catch {
            if fileManager.fileExists(atPath: stagingURL.path) {
                try? fileManager.removeItem(at: stagingURL)
            }
            logError("Import model directory failed: \(error)", category: .storage)
            throw StorageError.fileWriteFailed(error)
        }
    }

    /// Delete a downloaded model
    func deleteModel(_ modelId: String) async throws {
        let modelDir = modelPath(for: modelId)

        logDebug("[DELETE] Model ID: '\(modelId)'", category: .storage)
        logDebug("[DELETE] Computed path: \(modelDir.path)", category: .storage)
        logDebug("[DELETE] File exists: \(fileManager.fileExists(atPath: modelDir.path))", category: .storage)

        if fileManager.fileExists(atPath: modelDir.path) {
            do {
                try fileManager.removeItem(at: modelDir)

                // Verify deletion actually succeeded
                if fileManager.fileExists(atPath: modelDir.path) {
                    logError("[DELETE] FAILED - Directory still exists after removeItem!", category: .storage)
                    throw StorageError.fileWriteFailed(NSError(domain: "StorageService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Directory still exists after deletion"]))
                }

                logInfo("[DELETE] Successfully deleted model: \(modelId)", category: .storage)
            } catch {
                logError("[DELETE] removeItem failed: \(error)", category: .storage)
                throw StorageError.fileWriteFailed(error)
            }
        } else {
            logWarning("[DELETE] Directory not found at path: \(modelDir.path)", category: .storage)
        }
    }

    /// Get size of downloaded models
    func downloadedModelsSize() async -> Int64 {
        guard fileManager.fileExists(atPath: modelsDirectory.path) else { return 0 }

        var totalSize: Int64 = 0

        // Use contentsOfDirectory instead of enumerator to avoid async context issues
        func calculateSize(at url: URL) -> Int64 {
            var size: Int64 = 0
            if let contents = try? fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey]) {
                for item in contents {
                    if let values = try? item.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey]) {
                        if values.isDirectory == true {
                            size += calculateSize(at: item)
                        } else if let fileSize = values.fileSize {
                            size += Int64(fileSize)
                        }
                    }
                }
            }
            return size
        }

        totalSize = calculateSize(at: modelsDirectory)
        return totalSize
    }

    // MARK: - Settings

    /// Default values (used for Reset to Defaults and fresh install)
    static let defaultTemperatureValue: Float = 0.0
    static let defaultMaxTokensValue: Int = 2048
    static let defaultSystemPromptValue: String = "[DEFAULT_PROMPT]"  // Default Prompt - standard inference with no additional prompting
    static let defaultDebugLevelValue: Int = 0
    static let defaultRepetitionDetectionValue: Bool = false
    static let defaultAutoLoadLastModelValue: Bool = true
    static let defaultEnableMarkupValue: Bool = true
    static let defaultSendButtonOnLeftValue: Bool = false
    static let defaultLoadLastChatValue: Bool = false  // Don't load last chat on startup by default
    static let defaultLargeControlsValue: Bool = {
        // Default to large controls on iPad and visionOS for better touch targets
        #if os(visionOS)
        return true
        #elseif os(iOS)
        return UIDevice.current.userInterfaceIdiom == .pad  // Also true in iPad compat mode on visionOS
        #else
        return false
        #endif
    }()
    static let defaultShowMicrophoneValue: Bool = true  // Show microphone button by default

    // Sampling defaults
    static let defaultDoSampleValue: Bool = false  // Default: greedy (temperature=0)
    static let defaultTopPValue: Float = 0.95
    static let defaultTopKValue: Int = 0  // 0 = disabled
    static let defaultUseRecommendedSamplingValue: Bool = true  // Use model's recommended sampling if available

    /// Current settings (with defaults)
    var defaultTemperature: Float {
        UserDefaults.standard.object(forKey: "temperature") as? Float ?? Self.defaultTemperatureValue
    }

    var defaultMaxTokens: Int {
        UserDefaults.standard.object(forKey: "maxTokens") as? Int ?? Self.defaultMaxTokensValue
    }

    var defaultSystemPrompt: String {
        UserDefaults.standard.object(forKey: "systemPrompt") as? String ?? Self.defaultSystemPromptValue
    }

    var selectedModelId: String? {
        UserDefaults.standard.object(forKey: "selectedModelId") as? String
    }

    var autoLoadLastModel: Bool {
        UserDefaults.standard.object(forKey: "autoLoadLastModel") as? Bool ?? Self.defaultAutoLoadLastModelValue
    }

    func saveAutoLoadLastModel(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "autoLoadLastModel")
    }

    var debugLevel: Int {
        UserDefaults.standard.object(forKey: "debugLevel") as? Int ?? Self.defaultDebugLevelValue
    }

    func saveDebugLevel(_ value: Int) {
        UserDefaults.standard.set(value, forKey: "debugLevel")
    }

    var repetitionDetectionEnabled: Bool {
        UserDefaults.standard.object(forKey: "repetitionDetectionEnabled") as? Bool ?? Self.defaultRepetitionDetectionValue
    }

    func saveRepetitionDetectionEnabled(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "repetitionDetectionEnabled")
    }

    var enableMarkup: Bool {
        UserDefaults.standard.object(forKey: "enableMarkup") as? Bool ?? Self.defaultEnableMarkupValue
    }

    func saveEnableMarkup(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "enableMarkup")
    }

    var sendButtonOnLeft: Bool {
        UserDefaults.standard.object(forKey: "sendButtonOnLeft") as? Bool ?? Self.defaultSendButtonOnLeftValue
    }

    func saveSendButtonOnLeft(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "sendButtonOnLeft")
    }

    var loadLastChat: Bool {
        UserDefaults.standard.object(forKey: "loadLastChat") as? Bool ?? Self.defaultLoadLastChatValue
    }

    func saveLoadLastChat(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "loadLastChat")
    }

    var largeControls: Bool {
        UserDefaults.standard.object(forKey: "largeControls") as? Bool ?? Self.defaultLargeControlsValue
    }

    func saveLargeControls(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "largeControls")
    }

    var showMicrophone: Bool {
        UserDefaults.standard.object(forKey: "showMicrophone") as? Bool ?? Self.defaultShowMicrophoneValue
    }

    func saveShowMicrophone(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "showMicrophone")
    }

    // MARK: - Sampling Settings

    var doSample: Bool {
        UserDefaults.standard.object(forKey: "doSample") as? Bool ?? Self.defaultDoSampleValue
    }

    func saveDoSample(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "doSample")
    }

    var topP: Float {
        UserDefaults.standard.object(forKey: "topP") as? Float ?? Self.defaultTopPValue
    }

    func saveTopP(_ value: Float) {
        UserDefaults.standard.set(value, forKey: "topP")
    }

    var topK: Int {
        UserDefaults.standard.object(forKey: "topK") as? Int ?? Self.defaultTopKValue
    }

    func saveTopK(_ value: Int) {
        UserDefaults.standard.set(value, forKey: "topK")
    }

    var useRecommendedSampling: Bool {
        UserDefaults.standard.object(forKey: "useRecommendedSampling") as? Bool ?? Self.defaultUseRecommendedSamplingValue
    }

    func saveUseRecommendedSampling(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "useRecommendedSampling")
    }

    func clearLastModel() {
        UserDefaults.standard.removeObject(forKey: "selectedModelId")
    }

    func saveTemperature(_ value: Float) {
        UserDefaults.standard.set(value, forKey: "temperature")
    }

    func saveMaxTokens(_ value: Int) {
        UserDefaults.standard.set(value, forKey: "maxTokens")
    }

    func saveSystemPrompt(_ value: String) {
        UserDefaults.standard.set(value, forKey: "systemPrompt")
    }

    func saveSelectedModelId(_ value: String?) {
        UserDefaults.standard.set(value, forKey: "selectedModelId")
    }

    /// Reset all settings to defaults
    func resetToDefaults() {
        UserDefaults.standard.set(Self.defaultTemperatureValue, forKey: "temperature")
        UserDefaults.standard.set(Self.defaultMaxTokensValue, forKey: "maxTokens")
        UserDefaults.standard.set(Self.defaultSystemPromptValue, forKey: "systemPrompt")
        UserDefaults.standard.set(Self.defaultDebugLevelValue, forKey: "debugLevel")
        UserDefaults.standard.set(Self.defaultRepetitionDetectionValue, forKey: "repetitionDetectionEnabled")
        UserDefaults.standard.set(Self.defaultAutoLoadLastModelValue, forKey: "autoLoadLastModel")
        UserDefaults.standard.set(Self.defaultEnableMarkupValue, forKey: "enableMarkup")
        UserDefaults.standard.set(Self.defaultSendButtonOnLeftValue, forKey: "sendButtonOnLeft")
        UserDefaults.standard.set(Self.defaultLoadLastChatValue, forKey: "loadLastChat")
        UserDefaults.standard.set(Self.defaultLargeControlsValue, forKey: "largeControls")
        UserDefaults.standard.set(Self.defaultShowMicrophoneValue, forKey: "showMicrophone")
        UserDefaults.standard.set(Self.defaultDoSampleValue, forKey: "doSample")
        UserDefaults.standard.set(Self.defaultTopPValue, forKey: "topP")
        UserDefaults.standard.set(Self.defaultTopKValue, forKey: "topK")
        UserDefaults.standard.set(Self.defaultUseRecommendedSamplingValue, forKey: "useRecommendedSampling")
        logInfo("Settings reset to defaults", category: .storage)
    }
}
