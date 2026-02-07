//
//  StorageService.swift
//  ANEMLLChat
//
//  Persistence for conversations and settings
//

import Foundation

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

    private init() {
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        decoder.dateDecodingStrategy = .iso8601
    }

    // MARK: - Directories

    /// Documents directory URL (sandboxed on iOS, user's Documents on macOS)
    private var documentsDirectory: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// Conversations directory
    private var conversationsDirectory: URL {
        documentsDirectory.appendingPathComponent("Conversations", isDirectory: true)
    }

    /// Models directory (for downloaded models)
    /// - macOS: ~/Documents/ (directly in user's Documents for easy access)
    /// - iOS: Documents/Models/ (in app's sandboxed Documents)
    var modelsDirectory: URL {
        #if os(macOS)
        // Store directly in ~/Documents for easy Finder access (matches anemll-chatbot)
        return fileManager.homeDirectoryForCurrentUser.appendingPathComponent("Documents", isDirectory: true)
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
        documentsDirectory.appendingPathComponent("models.json")
    }

    /// Save custom models to registry
    func saveModelsRegistry(_ models: [ModelInfo]) async throws {
        // Only save custom models (not defaults)
        let customModels = models.filter { model in
            !ModelInfo.defaultModels.contains(where: { $0.id == model.id })
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
    static let defaultDebugDisablePrefillValue: Bool = false
    static let defaultDebugContextCapValue: Int = 0
    static let defaultDebugDisableIOBackingsValue: Bool = false
    static let defaultDebugRepeatInferCountValue: Int = 0
    static let defaultDebugRepeatOnlyDivergenceValue: Bool = false
    static let defaultDebugCompareKVStateEveryTokenValue: Bool = true
    static let defaultDebugPredictReadDelayMsValue: Double = 0.0
    static let defaultAutoLoadLastModelValue: Bool = true
    static let defaultEnableMarkupValue: Bool = true
    static let defaultSendButtonOnLeftValue: Bool = false
    static let defaultLoadLastChatValue: Bool = true  // Load last chat on startup by default
    static let defaultLargeControlsValue: Bool = false  // Large controls for accessibility
    static let defaultShowMicrophoneValue: Bool = true  // Show microphone button by default

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

    var debugDisablePrefill: Bool {
        UserDefaults.standard.object(forKey: "debugDisablePrefill") as? Bool ?? Self.defaultDebugDisablePrefillValue
    }

    func saveDebugDisablePrefill(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "debugDisablePrefill")
    }

    var debugContextCap: Int {
        UserDefaults.standard.object(forKey: "debugContextCap") as? Int ?? Self.defaultDebugContextCapValue
    }

    func saveDebugContextCap(_ value: Int) {
        UserDefaults.standard.set(value, forKey: "debugContextCap")
    }

    var debugDisableIOBackings: Bool {
        UserDefaults.standard.object(forKey: "debugDisableIOBackings") as? Bool ?? Self.defaultDebugDisableIOBackingsValue
    }

    func saveDebugDisableIOBackings(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "debugDisableIOBackings")
    }

    var debugRepeatInferCount: Int {
        UserDefaults.standard.object(forKey: "debugRepeatInferCount") as? Int ?? Self.defaultDebugRepeatInferCountValue
    }

    func saveDebugRepeatInferCount(_ value: Int) {
        UserDefaults.standard.set(value, forKey: "debugRepeatInferCount")
    }

    var debugRepeatOnlyDivergence: Bool {
        UserDefaults.standard.object(forKey: "debugRepeatOnlyDivergence") as? Bool ?? Self.defaultDebugRepeatOnlyDivergenceValue
    }

    func saveDebugRepeatOnlyDivergence(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "debugRepeatOnlyDivergence")
    }

    var debugCompareKVStateEveryToken: Bool {
        UserDefaults.standard.object(forKey: "debugCompareKVStateEveryToken") as? Bool ?? Self.defaultDebugCompareKVStateEveryTokenValue
    }

    func saveDebugCompareKVStateEveryToken(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "debugCompareKVStateEveryToken")
    }

    var debugPredictReadDelayMs: Double {
        if let value = UserDefaults.standard.object(forKey: "debugPredictReadDelayMs") as? Double {
            return max(0.0, min(value, 500.0))
        }

        // Backward compatibility with older int-based setting values.
        if let value = UserDefaults.standard.object(forKey: "debugPredictReadDelayMs") as? Int {
            return max(0.0, min(Double(value), 500.0))
        }

        return Self.defaultDebugPredictReadDelayMsValue
    }

    func saveDebugPredictReadDelayMs(_ value: Double) {
        UserDefaults.standard.set(max(0.0, min(value, 500.0)), forKey: "debugPredictReadDelayMs")
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
        UserDefaults.standard.set(Self.defaultDebugDisablePrefillValue, forKey: "debugDisablePrefill")
        UserDefaults.standard.set(Self.defaultDebugContextCapValue, forKey: "debugContextCap")
        UserDefaults.standard.set(Self.defaultDebugDisableIOBackingsValue, forKey: "debugDisableIOBackings")
        UserDefaults.standard.set(Self.defaultDebugRepeatInferCountValue, forKey: "debugRepeatInferCount")
        UserDefaults.standard.set(Self.defaultDebugRepeatOnlyDivergenceValue, forKey: "debugRepeatOnlyDivergence")
        UserDefaults.standard.set(Self.defaultDebugCompareKVStateEveryTokenValue, forKey: "debugCompareKVStateEveryToken")
        UserDefaults.standard.set(Self.defaultDebugPredictReadDelayMsValue, forKey: "debugPredictReadDelayMs")
        UserDefaults.standard.set(Self.defaultAutoLoadLastModelValue, forKey: "autoLoadLastModel")
        UserDefaults.standard.set(Self.defaultEnableMarkupValue, forKey: "enableMarkup")
        UserDefaults.standard.set(Self.defaultSendButtonOnLeftValue, forKey: "sendButtonOnLeft")
        UserDefaults.standard.set(Self.defaultLoadLastChatValue, forKey: "loadLastChat")
        UserDefaults.standard.set(Self.defaultLargeControlsValue, forKey: "largeControls")
        UserDefaults.standard.set(Self.defaultShowMicrophoneValue, forKey: "showMicrophone")
        logInfo("Settings reset to defaults", category: .storage)
    }
}
