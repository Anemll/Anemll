//
//  InferenceService.swift
//  ANEMLLChat
//
//  Wrapper around AnemllCore for model inference
//

import Foundation
import CoreML
@preconcurrency import AnemllCore

/// Errors during inference
enum InferenceError: LocalizedError {
    case modelNotLoaded
    case configNotFound
    case tokenizerFailed(Error)
    case modelLoadFailed(Error)
    case generationFailed(Error)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "No model is loaded"
        case .configNotFound: return "Model configuration not found"
        case .tokenizerFailed(let e): return "Tokenizer error: \(e.localizedDescription)"
        case .modelLoadFailed(let e): return "Model load error: \(e.localizedDescription)"
        case .generationFailed(let e): return "Generation error: \(e.localizedDescription)"
        case .cancelled: return "Generation was cancelled"
        }
    }
}

/// Result of token generation
struct GenerationResult: Sendable {
    let text: String
    let tokensPerSecond: Double
    let tokenCount: Int
    let windowShifts: Int
    let prefillTime: TimeInterval
    let prefillTokens: Int        // Number of input tokens (for prefill speed calculation)
    let historyTokens: Int        // Total history tokens (input + output) - matches CLI
    let isComplete: Bool
    let wasCancelled: Bool
    let stopReason: String
}

/// Loading progress information
struct ModelLoadingProgress: Sendable {
    let percentage: Double
    let stage: String
    let detail: String?
}

/// Thread-safe container for generation statistics
final class GenerationStats: @unchecked Sendable {
    var tokenCount: Int = 0
    var windowShifts: Int = 0
    var generatedText: String = ""
}

/// Repetition detector for detecting generation loops
final class RepetitionDetector: @unchecked Sendable {
    private var tokenHistory: [Int] = []
    private let windowSize: Int
    private let ngramSize: Int
    private let threshold: Int

    init(windowSize: Int = 50, ngramSize: Int = 5, threshold: Int = 3) {
        self.windowSize = windowSize
        self.ngramSize = ngramSize
        self.threshold = threshold
    }

    func addToken(_ token: Int) {
        tokenHistory.append(token)
        if tokenHistory.count > windowSize {
            tokenHistory.removeFirst()
        }
    }

    func isRepeating() -> Bool {
        guard tokenHistory.count >= ngramSize * threshold else { return false }

        // Check for repeated n-grams
        var ngramCounts: [String: Int] = [:]

        for i in 0...(tokenHistory.count - ngramSize) {
            let ngram = tokenHistory[i..<(i + ngramSize)].map(String.init).joined(separator: ",")
            ngramCounts[ngram, default: 0] += 1

            if ngramCounts[ngram]! >= threshold {
                return true
            }
        }

        return false
    }

    func reset() {
        tokenHistory.removeAll()
    }
}

/// Service for loading models and running inference
@MainActor
final class InferenceService: ObservableObject {
    static let shared = InferenceService()

    // State
    @Published private(set) var isModelLoaded = false
    @Published private(set) var isGenerating = false
    @Published private(set) var loadingProgress: ModelLoadingProgress?
    @Published private(set) var currentModelId: String?

    // Internal state
    private var config: YAMLConfig?
    private var tokenizer: Tokenizer?
    private var inferenceManager: InferenceManager?
    private var loadedModels: LoadedModels?

    // Generation control
    private var generationTask: Task<Void, Never>?
    private var shouldCancel = false
    private let repetitionDetector = RepetitionDetector()

    // Model template (detected from model path)
    private var currentTemplate: String = "default"

    // Settings
    var temperature: Float = 0.0  // Default: greedy decoding
    var maxTokens: Int = 512
    var systemPrompt: String = ""  // Default: no system prompt (matches CLI behavior)
    var debugLevel: Int = 0  // Debug verbosity: 0=off, 1=basic, 2=verbose
    var repetitionDetectionEnabled: Bool = false  // Default: off (matches CLI behavior)

    // Sampling settings
    var doSample: Bool = false  // If false, use greedy (temperature=0)
    var topP: Float = 0.95
    var topK: Int = 0  // 0 = disabled
    var useRecommendedSampling: Bool = true  // Use model's recommended sampling if available

    // Model's recommended sampling (set after model load)
    private(set) var modelRecommendedSampling: (doSample: Bool, temperature: Float, topP: Float, topK: Int)?
    private(set) var isArgmaxModel: Bool = false  // If true, sampling is unavailable

    /// Current model's maximum context size (stateLength or contextLength)
    /// Returns 2048 as default when no model is loaded
    var modelMaxContextSize: Int {
        guard let config = config else { return 2048 }
        // Use stateLength if available, otherwise contextLength
        return config.stateLength > 0 ? config.stateLength : config.contextLength
    }

    private init() {
        // Load settings from storage
        Task {
            temperature = await StorageService.shared.defaultTemperature
            maxTokens = await StorageService.shared.defaultMaxTokens
            systemPrompt = await StorageService.shared.defaultSystemPrompt
            debugLevel = await StorageService.shared.debugLevel
            repetitionDetectionEnabled = await StorageService.shared.repetitionDetectionEnabled
            doSample = await StorageService.shared.doSample
            topP = await StorageService.shared.topP
            topK = await StorageService.shared.topK
            useRecommendedSampling = await StorageService.shared.useRecommendedSampling
        }
    }

    // MARK: - Model Loading

    /// Load a model from a directory containing meta.yaml
    func loadModel(from modelPath: URL) async throws {
        let metaYamlPath = modelPath.appendingPathComponent("meta.yaml").path

        guard FileManager.default.fileExists(atPath: metaYamlPath) else {
            throw InferenceError.configNotFound
        }

        logInfo("Loading model from: \(modelPath.path)", category: .model)
        print("===== [MODEL LOADING] Starting to load model from: \(modelPath.path) =====")

        // Unload existing model
        await unloadModel()

        do {
            // Load configuration
            loadingProgress = ModelLoadingProgress(percentage: 0.05, stage: "Loading configuration", detail: nil)
            config = try YAMLConfig.load(from: metaYamlPath)

            guard let config = config else {
                throw InferenceError.configNotFound
            }

            // Load tokenizer
            loadingProgress = ModelLoadingProgress(percentage: 0.1, stage: "Loading tokenizer", detail: nil)
            let detectedTemplate = detectTemplate(from: config)
            currentTemplate = detectedTemplate
            tokenizer = try await Tokenizer(
                modelPath: modelPath.path,
                template: detectedTemplate,
                debugLevel: debugLevel
            )

            // Load models with progress
            loadingProgress = ModelLoadingProgress(percentage: 0.2, stage: "Loading CoreML models", detail: nil)

            let progressDelegate = LoadingProgressDelegate { [weak self] percentage, stage, detail in
                Task { @MainActor in
                    self?.loadingProgress = ModelLoadingProgress(
                        percentage: 0.2 + percentage * 0.7,
                        stage: stage,
                        detail: detail
                    )
                }
            }

            let modelLoader = ModelLoader(progressDelegate: progressDelegate)
            loadedModels = try await modelLoader.loadModel(from: config)

            // Create inference manager
            loadingProgress = ModelLoadingProgress(percentage: 0.95, stage: "Initializing inference engine", detail: nil)

            inferenceManager = try InferenceManager(
                models: loadedModels!,
                contextLength: config.contextLength,
                batchSize: config.batchSize,
                splitLMHead: config.splitLMHead,
                debugLevel: debugLevel,
                v110: config.configVersion == "0.1.1",
                argmaxInModel: config.argmaxInModel,
                slidingWindow: config.slidingWindow,
                updateMaskPrefill: config.updateMaskPrefill,
                prefillDynamicSlice: config.prefillDynamicSlice,
                modelPrefix: config.modelPrefix,
                vocabSize: config.vocabSize,
                lmHeadChunkSizes: config.lmHeadChunkSizes
            )
            print("===== [MODEL CONFIG] splitLMHead=\(config.splitLMHead), context=\(config.contextLength), batch=\(config.batchSize), argmax=\(config.argmaxInModel) =====")

            currentModelId = modelPath.lastPathComponent
            isModelLoaded = true
            loadingProgress = ModelLoadingProgress(percentage: 1.0, stage: "Ready", detail: nil)

            // Apply recommended sampling from model config
            applySamplingConfig(from: config)

            logInfo("Model loaded successfully", category: .model)
            print("===== [MODEL LOADED] Successfully loaded: \(modelPath.lastPathComponent) =====")

        } catch {
            loadingProgress = nil
            print("===== [MODEL ERROR] Failed to load model: \(error.localizedDescription) =====")
            throw InferenceError.modelLoadFailed(error)
        }
    }

    /// Unload the current model
    func unloadModel() async {
        cancelGeneration()

        inferenceManager?.unload()
        inferenceManager = nil
        loadedModels = nil
        tokenizer = nil
        config = nil
        currentModelId = nil
        isModelLoaded = false
        loadingProgress = nil

        logInfo("Model unloaded", category: .model)
    }

    // MARK: - Generation

    /// Generate a response for a conversation
    func generateResponse(
        for messages: [ChatMessage],
        onToken: @escaping @Sendable (String) -> Void,
        onWindowShift: @escaping @Sendable () -> Void,
        onHistoryUpdate: (@Sendable (Int) -> Void)? = nil  // Reports current historyTokens during generation
    ) async throws -> GenerationResult {
        guard let tokenizer = tokenizer,
              let inferenceManager = inferenceManager else {
            throw InferenceError.modelNotLoaded
        }

        isGenerating = true
        shouldCancel = false
        repetitionDetector.reset()

        defer {
            isGenerating = false
        }

        let capturedTemplate = currentTemplate
        let capturedSystemPrompt = systemPrompt
        let capturedConfig = config
        let capturedTokenizerForPrep = tokenizer

        struct PreparedInput {
            let chatMessages: [Tokenizer.ChatMessage]
            let inputTokens: [Int]
            let contextLength: Int
            let maxContextSize: Int
            let historyTrimmed: Bool
            let originalSize: Int
        }

        let prepared = await Task.detached(priority: .userInitiated) {
            () -> PreparedInput in
            // Convert messages to tokenizer format
            var chatMessages: [Tokenizer.ChatMessage] = []

            // Add system prompt if not present (resolve markers to actual prompts)
            if !messages.contains(where: { $0.role == .system }) {
                let resolvedPrompt = InferenceService.resolveSystemPrompt(
                    capturedSystemPrompt,
                    template: capturedTemplate
                )
                if !resolvedPrompt.isEmpty {
                    chatMessages.append(.system(resolvedPrompt))
                }
            }

            for message in messages {
                switch message.role {
                case .system:
                    chatMessages.append(.system(message.content))
                case .user:
                    chatMessages.append(.user(message.content))
                case .assistant:
                    chatMessages.append(.assistant(message.content))
                }
            }

            // Get context limits (match CLI: use stateLength if available, otherwise contextLength)
            let contextLength = capturedConfig?.contextLength ?? 512
            let stateLength = (capturedConfig?.stateLength ?? 0) > 0 ? (capturedConfig?.stateLength ?? contextLength) : contextLength
            let maxContextSize = stateLength - 100  // Leave room for response (match CLI/Python)

            // Tokenize and check size
            var inputTokens = capturedTokenizerForPrep.applyChatTemplate(
                input: chatMessages,
                addGenerationPrompt: true
            )

            // Trim history if exceeds context (match CLI behavior)
            let originalSize = inputTokens.count
            var historyTrimmed = false
            while inputTokens.count > maxContextSize && chatMessages.count > 2 {
                historyTrimmed = true
                // Remove oldest message pair (user + assistant) like CLI does
                // Skip system prompt if present (index 0)
                let startIndex = (chatMessages.first?.role == "system") ? 1 : 0
                if chatMessages.count > startIndex + 2 {
                    chatMessages.remove(at: startIndex)  // Remove oldest user
                    if chatMessages.count > startIndex {
                        chatMessages.remove(at: startIndex)  // Remove oldest assistant
                    }
                    inputTokens = capturedTokenizerForPrep.applyChatTemplate(
                        input: chatMessages,
                        addGenerationPrompt: true
                    )
                } else {
                    break
                }
            }

            return PreparedInput(
                chatMessages: chatMessages,
                inputTokens: inputTokens,
                contextLength: contextLength,
                maxContextSize: maxContextSize,
                historyTrimmed: historyTrimmed,
                originalSize: originalSize
            )
        }.value

        let chatMessages = prepared.chatMessages
        let inputTokens = prepared.inputTokens
        let contextLength = prepared.contextLength
        let maxContextSize = prepared.maxContextSize

        if !messages.contains(where: { $0.role == .system }) {
            let resolvedPrompt = InferenceService.resolveSystemPrompt(
                capturedSystemPrompt,
                template: capturedTemplate
            )
            print("===== [SYSTEM PROMPT] Template: \(capturedTemplate), Raw: '\(capturedSystemPrompt)', Resolved: '\(resolvedPrompt)' =====")
            logInfo("System prompt resolved: template=\(capturedTemplate), prompt=\(resolvedPrompt)", category: .inference)
        }

        if prepared.historyTrimmed {
            print("[SYSTEM] History trimmed: \(prepared.originalSize) → \(inputTokens.count) tokens, \(chatMessages.count) msgs remaining")
        }

        // Debug: print messages being sent to tokenizer
        print("===== [CHAT MESSAGES] \(chatMessages.count) messages being tokenized =====")
        for (i, msg) in chatMessages.enumerated() {
            let role = msg.isUser ? "user" : (msg.isAssistant ? "assistant" : "system")
            let preview = msg.content.prefix(50)
            print("  [\(i)] \(role): \(preview)\(msg.content.count > 50 ? "..." : "")")
        }

        print("===== [INFERENCE] Input tokens: \(inputTokens.count) / Context: \(contextLength) (max: \(maxContextSize)) =====")
        if inputTokens.count > contextLength {
            print("⚠️  WARNING: Input tokens (\(inputTokens.count)) exceed context length (\(contextLength))!")
        }
        logDebug("Input tokens: \(inputTokens.count)", category: .inference)

        // Track statistics using thread-safe container
        let stats = GenerationStats()
        let startTime = Date()

        // Capture values we need in detached task
        let capturedRepetitionDetector = repetitionDetector
        let capturedInferenceManager = inferenceManager
        let capturedRepetitionEnabled = repetitionDetectionEnabled
        let capturedTemperature = temperature
        let capturedMaxTokens = maxTokens
        let capturedTokenizer = tokenizer
        let capturedDebugLevel = debugLevel
        let inputTokenCount = inputTokens.count  // Capture for history calculation

        do {
            let (prefillTime, stopReason) = try await Task.detached(priority: .userInitiated) {
                () async throws -> (TimeInterval, String) in
                var pendingText = ""
                var lastEmitTime = CFAbsoluteTimeGetCurrent()
                var lastHistoryEmitTime = lastEmitTime
                let emitInterval: CFAbsoluteTime = 1.0 / 30.0
                let minChunkChars = 48
                var lastTokenLogTime = CFAbsoluteTimeGetCurrent()
                var lastTokenCount = 0
                // Full-sequence decode state to preserve SentencePiece spaces
                var allTokenIds: [Int] = []
                var prevDecodedText: String = ""

                if capturedDebugLevel >= 2 {
                    logDebug("[Inference] start detached", category: .inference)
                }

                func emitIfNeeded(force: Bool = false) {
                    let now = CFAbsoluteTimeGetCurrent()
                    let shouldEmit = force || now - lastEmitTime >= emitInterval || pendingText.count >= minChunkChars

                    if shouldEmit, !pendingText.isEmpty {
                        let chunk = pendingText
                        pendingText = ""
                        lastEmitTime = now
                        onToken(chunk)
                    }

                    if let onHistoryUpdate, force || now - lastHistoryEmitTime >= emitInterval {
                        lastHistoryEmitTime = now
                        onHistoryUpdate(inputTokenCount + stats.tokenCount)
                    }
                }

                let (_, prefillTime, stopReason) = try await capturedInferenceManager.generateResponse(
                    initialTokens: inputTokens,
                    temperature: capturedTemperature,
                    maxTokens: capturedMaxTokens,
                    eosTokens: capturedTokenizer.eosTokenIds,
                    tokenizer: capturedTokenizer,
                    onToken: { token in
                        // Check for repetition only if enabled
                        if capturedRepetitionEnabled {
                            capturedRepetitionDetector.addToken(token)
                            if capturedRepetitionDetector.isRepeating() {
                                logWarning("Repetition detected, aborting", category: .inference)
                                capturedInferenceManager.AbortGeneration(Code: 2)
                                return
                            }
                        }

                        stats.tokenCount += 1
                        // Full-sequence decode + diff to preserve SentencePiece spaces
                        allTokenIds.append(token)
                        let fullText = capturedTokenizer.decode(tokens: allTokenIds)
                        let text: String
                        if fullText.count > prevDecodedText.count {
                            text = String(fullText[fullText.index(fullText.startIndex, offsetBy: prevDecodedText.count)...])
                        } else {
                            text = capturedTokenizer.decode(tokens: [token])
                        }
                        prevDecodedText = fullText
                        stats.generatedText += text

                        pendingText += text
                        emitIfNeeded()

                        if capturedDebugLevel >= 2 {
                            let now = CFAbsoluteTimeGetCurrent()
                            if now - lastTokenLogTime >= 1.0 {
                                let delta = stats.tokenCount - lastTokenCount
                                lastTokenCount = stats.tokenCount
                                lastTokenLogTime = now
                                logDebug("[Decode] +\(delta) tok/s pendingChars=\(pendingText.count)", category: .inference)
                            }
                        }
                    },
                    onWindowShift: {
                        stats.windowShifts += 1
                        onWindowShift()
                    }
                )
                emitIfNeeded(force: true)
                return (prefillTime, stopReason)
            }.value

            if capturedDebugLevel >= 2 {
                logDebug("[Inference] prefill=\(String(format: "%.3f", prefillTime))s stop=\(stopReason)", category: .inference)
            }

            let totalTime = Date().timeIntervalSince(startTime)
            let tokensPerSecond = totalTime > 0 ? Double(stats.tokenCount) / totalTime : 0

            logInfo("Generation complete: \(stats.tokenCount) tokens, \(String(format: "%.1f", tokensPerSecond)) tok/s", category: .inference)
            print("===== [INFERENCE] Complete: \(stats.tokenCount) tokens at \(String(format: "%.1f", tokensPerSecond)) tok/s =====")

            // Total history tokens = input (prefill) + output (generated) - matches CLI behavior
            let historyTokens = inputTokens.count + stats.tokenCount

            let wasCancelled = shouldCancel || stopReason.hasPrefix("abort_generation1")

            return GenerationResult(
                text: stats.generatedText,
                tokensPerSecond: tokensPerSecond,
                tokenCount: stats.tokenCount,
                windowShifts: stats.windowShifts,
                prefillTime: prefillTime,
                prefillTokens: inputTokens.count,  // For prefill speed calculation
                historyTokens: historyTokens,       // Total history like CLI shows
                isComplete: true,
                wasCancelled: wasCancelled,
                stopReason: stopReason
            )

        } catch {
            if shouldCancel {
                let historyTokens = inputTokens.count + stats.tokenCount
                return GenerationResult(
                    text: stats.generatedText,
                    tokensPerSecond: 0,
                    tokenCount: stats.tokenCount,
                    windowShifts: stats.windowShifts,
                    prefillTime: 0,
                    prefillTokens: inputTokens.count,  // For prefill speed calculation
                    historyTokens: historyTokens,       // Total history like CLI shows
                    isComplete: false,
                    wasCancelled: true,
                    stopReason: "cancelled"
                )
            }
            throw InferenceError.generationFailed(error)
        }
    }

    /// Cancel ongoing generation
    func cancelGeneration() {
        shouldCancel = true
        inferenceManager?.AbortGeneration(Code: 1)
        generationTask?.cancel()
        generationTask = nil
    }

    // MARK: - Sampling Configuration

    /// Apply recommended sampling settings from model config
    private func applySamplingConfig(from config: YAMLConfig) {
        // Check for argmax model
        isArgmaxModel = config.argmaxInModel
        if isArgmaxModel {
            logWarning("Model uses argmax output - sampling is unavailable, using greedy decoding", category: .model)
            print("===== [SAMPLING] Argmax model detected - sampling unavailable, using greedy =====")
            modelRecommendedSampling = nil
            return
        }

        // Extract recommended sampling from config
        if let rec = config.recommendedSampling {
            modelRecommendedSampling = (
                doSample: rec.doSample,
                temperature: Float(rec.temperature),
                topP: Float(rec.topP),
                topK: rec.topK
            )

            print("===== [SAMPLING] Model recommends: doSample=\(rec.doSample), temp=\(rec.temperature), topP=\(rec.topP), topK=\(rec.topK) =====")
            logInfo("Model recommended sampling: temp=\(rec.temperature), topP=\(rec.topP), topK=\(rec.topK)", category: .model)

            // Apply if user enabled "use recommended sampling"
            if useRecommendedSampling {
                doSample = rec.doSample
                temperature = Float(rec.temperature)
                topP = Float(rec.topP)
                topK = rec.topK
                logInfo("Applied model-recommended sampling settings", category: .model)
                print("===== [SAMPLING] Applied recommended settings =====")
            }
        } else {
            modelRecommendedSampling = nil
            logDebug("No recommended sampling in model config", category: .model)
        }
    }

    /// Get effective sampling parameters (respects argmax, recommended, and user settings)
    var effectiveSamplingDescription: String {
        if isArgmaxModel {
            return "Greedy (argmax model)"
        }

        if useRecommendedSampling, let rec = modelRecommendedSampling {
            return "Model recommended: \(String(format: "%.2f", rec.temperature)) / \(String(format: "%.2f", rec.topP)) / \(rec.topK)"
        }

        if doSample && temperature > 0 {
            return "Custom: \(String(format: "%.2f", temperature)) / \(String(format: "%.2f", topP)) / \(topK)"
        }

        return "Greedy"
    }

    // MARK: - Helpers

    /// Resolve system prompt marker to actual prompt based on model template
    nonisolated private static func resolveSystemPrompt(_ prompt: String, template: String) -> String {
        let normalizedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)

        // Default / no-template markers map to "no extra system prompt".
        if normalizedPrompt.isEmpty || normalizedPrompt == "[DEFAULT_PROMPT]" || normalizedPrompt == "[NO_TEMPLATE]" {
            return ""
        }

        // If this is not a model marker, treat it as a custom prompt.
        guard normalizedPrompt.hasPrefix("[MODEL_") else {
            return normalizedPrompt
        }

        // Define default prompts for each model type
        let defaultPrompts: [String: String] = [
            "gemma": "You are a helpful assistant.",
            "gemma3": "You are a helpful assistant.",
            "llama": "You are a helpful, respectful and honest assistant.",
            "llama3": "You are a helpful, respectful and honest assistant.",
            "qwen": "You are Qwen, a helpful assistant.",
            "qwen3": "You are Qwen, a helpful assistant.",
            "deepseek": "You are a helpful assistant.",
            "default": "You are a helpful assistant."
        ]

        // Thinking mode prompts (Qwen3 supports /think and /no_think)
        let thinkingPrompts: [String: String] = [
            "qwen": "You are Qwen, a helpful assistant. /think",
            "qwen3": "You are Qwen, a helpful assistant. /think",
            "gemma3": "You are a helpful assistant.",  // No explicit thinking mode
            "llama3": "You are a helpful, respectful and honest assistant.",  // No explicit thinking mode
            "default": "You are a helpful assistant."
        ]

        // Non-thinking mode prompts
        let nonThinkingPrompts: [String: String] = [
            "qwen": "You are Qwen, a helpful assistant. /no_think",
            "qwen3": "You are Qwen, a helpful assistant. /no_think",
            "gemma3": "You are a helpful assistant.",
            "llama3": "You are a helpful, respectful and honest assistant.",
            "default": "You are a helpful assistant."
        ]

        let template = template.lowercased()

        switch normalizedPrompt {
        case "[MODEL_DEFAULT]":
            return defaultPrompts[template] ?? defaultPrompts["default"]!
        case "[MODEL_THINKING]":
            return thinkingPrompts[template] ?? thinkingPrompts["default"]!
        case "[MODEL_NON_THINKING]":
            return nonThinkingPrompts[template] ?? nonThinkingPrompts["default"]!
        default:
            return prompt
        }
    }

    private func detectTemplate(from config: YAMLConfig) -> String {
        // Detect template from model path or config
        let path = config.modelPath.lowercased()

        if path.contains("gemma") {
            return path.contains("gemma3") ? "gemma3" : "gemma"
        } else if path.contains("qwen") {
            return "qwen"
        } else if path.contains("deepseek") {
            return "deepseek"
        } else if path.contains("deephermes") {
            return "deephermes"
        } else if path.contains("llama") {
            return "llama3"
        }

        return "default"
    }
}

// MARK: - Loading Progress Delegate

private final class LoadingProgressDelegate: ModelLoadingProgressDelegate, @unchecked Sendable {
    private let onProgress: (Double, String, String?) -> Void

    init(onProgress: @escaping (Double, String, String?) -> Void) {
        self.onProgress = onProgress
    }

    func loadingProgress(percentage: Double, stage: String, detail: String?) {
        onProgress(percentage, stage, detail)
    }

    func loadingCompleted(models: LoadedModels) {
        onProgress(1.0, "Complete", nil)
    }

    func loadingCancelled() {
        onProgress(0, "Cancelled", nil)
    }

    func loadingFailed(error: Error) {
        onProgress(0, "Failed", error.localizedDescription)
    }
}
