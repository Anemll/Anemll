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

    // Settings
    var temperature: Float = 0.7
    var maxTokens: Int = 512
    var systemPrompt: String = "You are a helpful assistant."

    private init() {
        // Load settings from storage
        Task {
            temperature = await StorageService.shared.defaultTemperature
            maxTokens = await StorageService.shared.defaultMaxTokens
            systemPrompt = await StorageService.shared.defaultSystemPrompt
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
            tokenizer = try await Tokenizer(
                modelPath: modelPath.path,
                template: detectTemplate(from: config),
                debugLevel: 0
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
                debugLevel: 0,
                argmaxInModel: config.argmaxInModel,
                slidingWindow: config.slidingWindow
            )
            print("===== [MODEL CONFIG] splitLMHead=\(config.splitLMHead), context=\(config.contextLength), batch=\(config.batchSize), argmax=\(config.argmaxInModel) =====")

            currentModelId = modelPath.lastPathComponent
            isModelLoaded = true
            loadingProgress = ModelLoadingProgress(percentage: 1.0, stage: "Ready", detail: nil)

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
        onWindowShift: @escaping @Sendable () -> Void
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

        // Convert messages to tokenizer format
        var chatMessages: [Tokenizer.ChatMessage] = []

        // Add system prompt if not present
        if !messages.contains(where: { $0.role == .system }) {
            chatMessages.append(.system(systemPrompt))
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

        // Tokenize with chat template
        let inputTokens = tokenizer.applyChatTemplate(
            input: chatMessages,
            addGenerationPrompt: true
        )

        logDebug("Input tokens: \(inputTokens.count)", category: .inference)
        print("===== [INFERENCE] Starting generation with \(inputTokens.count) input tokens =====")

        // Track statistics using thread-safe container
        let stats = GenerationStats()
        let startTime = Date()

        // Capture values we need in nonisolated closures
        let capturedRepetitionDetector = repetitionDetector
        let capturedInferenceManager = inferenceManager

        do {
            let (_, prefillTime, stopReason) = try await inferenceManager.generateResponse(
                initialTokens: inputTokens,
                temperature: temperature,
                maxTokens: maxTokens,
                eosTokens: tokenizer.eosTokenIds,
                tokenizer: tokenizer,
                onToken: { token in
                    // Check for repetition (non-isolated access)
                    capturedRepetitionDetector.addToken(token)
                    if capturedRepetitionDetector.isRepeating() {
                        logWarning("Repetition detected, aborting", category: .inference)
                        capturedInferenceManager.AbortGeneration(Code: 2)
                        return
                    }

                    stats.tokenCount += 1
                    let text = tokenizer.decode(tokens: [token])
                    stats.generatedText += text

                    onToken(text)
                },
                onWindowShift: {
                    stats.windowShifts += 1
                    onWindowShift()
                }
            )

            let totalTime = Date().timeIntervalSince(startTime)
            let tokensPerSecond = totalTime > 0 ? Double(stats.tokenCount) / totalTime : 0

            logInfo("Generation complete: \(stats.tokenCount) tokens, \(String(format: "%.1f", tokensPerSecond)) tok/s", category: .inference)
            print("===== [INFERENCE] Complete: \(stats.tokenCount) tokens at \(String(format: "%.1f", tokensPerSecond)) tok/s =====")

            return GenerationResult(
                text: stats.generatedText,
                tokensPerSecond: tokensPerSecond,
                tokenCount: stats.tokenCount,
                windowShifts: stats.windowShifts,
                prefillTime: prefillTime,
                isComplete: true,
                wasCancelled: shouldCancel,
                stopReason: stopReason
            )

        } catch {
            if shouldCancel {
                return GenerationResult(
                    text: stats.generatedText,
                    tokensPerSecond: 0,
                    tokenCount: stats.tokenCount,
                    windowShifts: stats.windowShifts,
                    prefillTime: 0,
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

    // MARK: - Helpers

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
