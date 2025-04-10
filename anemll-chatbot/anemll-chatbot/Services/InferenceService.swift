// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// InferenceService.swift

import Foundation
import CoreML
import Combine
import Yams
import AnemllCore  // Import AnemllCore for InferenceManager, Tokenizer, etc.

// Add constants for abort reasons
private enum AbortReason {
    static let userCancelled = 1
    static let repetitionDetected = 2
}

// StandardOutputObserver to intercept standard output if needed
final class StandardOutputObserver {
    private var pipe = Pipe()
    private var originalStdout: Int32 = -1
    private var readDataTask: Task<Void, Error>? = nil
    
    func start(messageHandler: @escaping (String) -> Void) {
        // Save original stdout
        originalStdout = dup(STDOUT_FILENO)
        
        // Redirect stdout to our pipe
        setvbuf(stdout, nil, _IONBF, 0)
        dup2(pipe.fileHandleForWriting.fileDescriptor, STDOUT_FILENO)
        
        // Start reading from the pipe
        readDataTask = Task {
            for try await line in pipe.fileHandleForReading.bytes.lines {
                // Process each line of output
                messageHandler(line)
                
                // Write the output back to original stdout
                if let data = (line + "\n").data(using: .utf8) {
                    // Use withUnsafeBytes to convert Data to UnsafeRawPointer
                    data.withUnsafeBytes { buffer in
                        _ = write(originalStdout, buffer.baseAddress, buffer.count)
                    }
                }
            }
        }
    }
    
    func stop() {
        // Cancel the read task
        readDataTask?.cancel()
        
        // Restore original stdout
        if originalStdout != -1 {
            dup2(originalStdout, STDOUT_FILENO)
            close(originalStdout)
            originalStdout = -1
        }
    }
    
    deinit {
        stop()
    }
}

enum InferenceError: Error {
    case modelNotLoaded
    case invalidConfig
    case tokenizationFailed
    case inferenceError(String)
    case modelPathNotFound
    case contextTooLong
}

// TokenPrinter is now in a separate file

// Add a struct to hold inference results
struct InferenceResult {
    let text: String
    let tokensPerSecond: Double
    let tokenCount: Int
    let windowShifts: Int
    let isComplete: Bool // Whether this is the final result
    var wasCancelled: Bool = false // Whether generation was cancelled
}

// Add TokenBuffer class after the InferenceResult struct
/// A buffer to store and manage tokens from long-form generation
class TokenBuffer {
    private var tokens: [Int] = []
    private var text: String = ""
    private var tokenizer: Tokenizer
    private var windowShiftCount: Int = 0
    
    init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }
    
    func addToken(_ token: Int) {
        tokens.append(token)
        // Decode incrementally to maintain text representation
        text = tokenizer.decode(tokens: tokens)
    }
    
    func recordWindowShift() {
        windowShiftCount += 1
    }
    
    func clear() {
        tokens = []
        text = ""
        windowShiftCount = 0
    }
    
    func getTokens() -> [Int] {
        return tokens
    }
    
    func getText() -> String {
        return text
    }
    
    func getWindowShifts() -> Int {
        return windowShiftCount
    }
}

// Add the RepetitionDetector class before InferenceService
class RepetitionDetector {
    private let windowSize: Int
    private let ngramSize: Int
    private let threshold: Int
    private var tokenWindow: [Int] = []

    init(windowSize: Int = 50, ngramSize: Int = 5, threshold: Int = 3) {
        self.windowSize = windowSize
        self.ngramSize = ngramSize
        self.threshold = threshold
    }

    func addToken(_ token: Int) -> Bool {
        tokenWindow.append(token)
        if tokenWindow.count > windowSize {
            tokenWindow.removeFirst()
        }
        if tokenWindow.count >= ngramSize {
            var counts: [[Int]: Int] = [:]
            for i in 0..<(tokenWindow.count - ngramSize + 1) {
                let ngram = Array(tokenWindow[i..<(i + ngramSize)])
                counts[ngram, default: 0] += 1
                if counts[ngram]! > threshold {
                    return true
                }
            }
        }
        return false
    }
}

@MainActor
class InferenceService: ObservableObject, ModelLoadingProgressDelegate {
    // Use a properly isolated shared instance that is created on the main actor
    static let shared: InferenceService = {
        // This initializer runs on the main actor since the class is @MainActor-isolated
        return InferenceService()
    }()
    
    private var inferenceManager: InferenceManager?
    private var tokenizer: Tokenizer?
    private var currentModelId: String?
    private let defaultTemperature: Float = 0.6
    
    // Add a property to store the context length from config
    private var modelContextLength: Int = 2048
    
    // Update debug level property to default to 2 for detailed hidden states debugging
    private var debugLevel: Int = 0
    
    // Add conversation state tracking
    private var currentState: ConversationState?
    
    // Track active inference generation tasks to allow cancellation
    private var activeInferenceTask: Task<Void, Error>?
    // Make the cancellation flag nonisolated so it can be safely accessed from any context
    nonisolated(unsafe) var inferenceIsCancelled = false
    
    // Add a struct to track conversation state
    struct ConversationState {
        var chatId: String
        var currentPosition: Int = 0
        var tokenCount: Int = 0
        var isNewChat: Bool = true
        var lastMessageTimestamp: Date = Date()
        
        // For tracking KV cache in future implementations
        // var kvCacheState: Any?
    }
    
    // Add a cancellation flag and task
    private var modelLoadingTask: Task<Void, Error>?
    private var isCancelled = false
    private var suppressInterruptionNotification = false
    
    // Add a reference to the current model loader for proper cancellation
    private var currentModelLoader: ModelLoader?
    
    // Loading progress tracking
    @Published var totalComponents: Int = 0
    @Published var loadedComponents: Int = 0
    
    // Add a dedicated cancellation reason
    enum CancellationReason {
        case userInitiated
        case startingNewModel
    }
    
    // Track the last cancellation reason - making it main actor isolated
    @MainActor
    private var lastCancellationReason: CancellationReason = .userInitiated
    
    // Add a property to safely access the cancellation reason from nonisolated contexts
    nonisolated(unsafe) private var _cancellationReasonForDelegate: CancellationReason = .userInitiated
    
    // Add warmup configuration
    private var warmupOnLoad: Bool = false
    private var testOnLoad: Bool = true

    // Add published property for last loading error
    @Published var lastLoadingError: String?
    
    // Add array to track all loading errors in current attempt
    @Published private var loadingErrors: [String] = []
    
    // Add property to track if we're in a loading attempt
    private var isInLoadingAttempt: Bool = false
    
    /// Get all errors from the current/last loading attempt
    var currentLoadingErrors: [String] {
        return loadingErrors
    }
    
    // Add repetition detector configuration
    private var useRepetitionDetector: Bool = true
    
    /// Configure whether to use repetition detection during inference
    func configureRepetitionDetector(enabled: Bool) {
        print("🔧 Configuring repetition detector: \(enabled ? "enabled" : "disabled")")
        useRepetitionDetector = enabled
    }
    
    /// Get the current state of the repetition detector
    var isRepetitionDetectorEnabled: Bool {
        useRepetitionDetector
    }
    
    // MARK: - Published Properties with Observers
    
    @Published var isModelLoaded: Bool = false {
        didSet {
            print("📊 PUBLISHED PROPERTY: isModelLoaded changed to \(isModelLoaded)")
            // Also notify about the changed property
            NotificationCenter.default.post(
                name: Notification.Name("ModelLoadedChanged"),
                object: isModelLoaded
            )
        }
    }
    
    @Published var loadingProgress: Double = 0.0 {
        didSet {
            let percentage = Int(loadingProgress * 100)
            print("📊 PUBLISHED PROPERTY: loadingProgress changed to \(percentage)%")
            
            // Synchronize the string representation
            loadingProgressString = "\(percentage)%"
            
            // Notify about the changed progress
            NotificationCenter.default.post(
                name: Notification.Name("LoadingProgressChanged"),
                object: loadingProgress
            )
        }
    }
    
    @Published var loadingProgressString: String = "0%" {
        didSet {
            print("📊 PUBLISHED PROPERTY: loadingProgressString changed to \(loadingProgressString)")
        }
    }
    
    @Published var isLoadingModel: Bool = false {
        didSet {
            print("📊 PUBLISHED PROPERTY: isLoadingModel changed to \(isLoadingModel)")
        }
    }
    
    @Published var loadingStatus: String = "" {
        didSet {
            print("📊 PUBLISHED PROPERTY: loadingStatus changed to \(loadingStatus)")
        }
    }
    
    @Published var hasLoadingError: Bool = false // Track if loading had an error
    
    // MARK: - ModelLoadingProgressDelegate Implementation
    
    nonisolated func loadingProgress(percentage: Double, stage: String, detail: String?) {
        // Use the status format from before
        let status = detail != nil ? "\(stage): \(detail!)" : stage
        print("📊 MODEL LOADING: \(stage) - \(Int(percentage * 100))% - \(detail ?? "")")
        
        // Use Task to switch to the main actor for UI updates
        Task { @MainActor in
            // First, directly update the progress bar for immediate feedback
            self.updateLoadingProgress(percentage)
            
            // Then update the rest of the UI
            self.updateUI(status: status)
            
            // Force UI to refresh to ensure changes are visible
            self.forceUIRefresh()
        }
    }
    
    nonisolated func loadingCancelled() {
        print("⛔️ DELEGATE: Model loading cancelled, reason: \(self._cancellationReasonForDelegate)")
        
        Task { @MainActor in
            print("⛔️ DELEGATE: Inside Task in loadingCancelled(), updating UI")
            
            // Add cancellation message to loading errors
            self.loadingErrors.append("Model loading cancelled")
            
            // Use our helper to update the UI
            self.updateUI(
                progress: 0,
                status: "Loading cancelled",
                isLoading: false,
                isLoaded: false
            )
            
            // Set error flag
            self.hasLoadingError = true
            
            // Only post notification if this was a user-initiated cancellation
            if self._cancellationReasonForDelegate == .userInitiated {
                if let modelId = self.currentModelId {
                    print("⛔️ DELEGATE: Posting ModelLoadingInterrupted notification - user initiated")
                    NotificationCenter.default.post(
                        name: Notification.Name("ModelLoadingInterrupted"),
                        object: modelId
                    )
                    print("📣 Posted ModelLoadingInterrupted notification - user initiated")
                }
            } else {
                print("⛔️ DELEGATE: Suppressed ModelLoadingInterrupted notification - starting new model")
                print("🔕 Suppressed ModelLoadingInterrupted notification - starting new model")
            }
            
            // Force UI to refresh to ensure changes are visible
            self.forceUIRefresh()
            
            // Reset any task reference to ensure clean state
            print("⛔️ DELEGATE: Resetting modelLoadingTask to nil")
            self.modelLoadingTask = nil
            
            // Ensure isCancelled flag is reset after a short delay
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                print("⛔️ DELEGATE: Resetting isCancelled flag to false")
                self?.isCancelled = false
            }
        }
    }
    
        
    // Add warmup inference method
    private func performTestInference() async throws {
        print("🔥 Performing performTestInference inference...")
        guard let inferenceManager = inferenceManager, let tokenizer = tokenizer else {
            print("⚠️ Cannot perform warmup: inferenceManager or tokenizer is nil")
            return
        }
        
        let maxTokens = 5
        let TestText = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        Cutting Knowledge Date: December 2023
        Today Date: 26 Jul 2024

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>

        Thinking mode disabled<|eot_id|><|start_header_id|>user<|end_header_id|>

        Who are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        I'm an artificial intelligence model that provides general information and discussion on a variety of topics, including but not limited to history, science, technology, culture, and many other subjects.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Who made you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        
        do {
            // Tokenize the warmup text
            let tokens = tokenizer.tokenize(TestText)
            
            // Perform inference using generateResponse
            for _ in 1...2 {
                let (_, _, _) = try await inferenceManager.generateResponse(
                    initialTokens: tokens,
                    temperature: defaultTemperature,
                    maxTokens: maxTokens,
                    eosToken: tokenizer.eosTokenId,
                    tokenizer: tokenizer,
                    onToken: { token in
                        print("🔥 TEST: Generated token \(token)")
                    }
                )
            }
            print("✅ test inference completed successfully")
            //print("   Prefill time: \(String(format: "%.3f", prefillTime))s")
            //print("   Stop reason: \(stopReason)")
        } catch {
            print("⚠️ Warmup inference failed: \(error)")
            // Don't throw the error as warmup failure shouldn't block model loading
        }
    }

    
    
    // Add warmup inference method
    private func performWarmupInference() async throws {
        print("🔥 Performing warmup inference...")
        guard let inferenceManager = inferenceManager, let tokenizer = tokenizer else {
            print("⚠️ Cannot perform warmup: inferenceManager or tokenizer is nil")
            return
        }
        
        let warmupText = "who are you"
        let maxTokens = 12
        
        do {
            // Tokenize the warmup text
            let tokens = tokenizer.tokenize(warmupText)
            
            // Perform inference using generateResponse
            let (_, prefillTime, stopReason) = try await inferenceManager.generateResponse(
                initialTokens: tokens,
                temperature: defaultTemperature,
                maxTokens: maxTokens,
                eosToken: tokenizer.eosTokenId,
                tokenizer: tokenizer,
                onToken: { token in
                    // Just log every 5 tokens for monitoring
                    if token % 5 == 0 {
                        print("🔥 WARMUP: Generated token \(token)")
                    }
                }
            )
            print("✅ Warmup inference completed successfully")
            print("   Prefill time: \(String(format: "%.3f", prefillTime))s")
            print("   Stop reason: \(stopReason)")
        } catch {
            print("⚠️ Warmup inference failed: \(error)")
            // Don't throw the error as warmup failure shouldn't block model loading
        }
    }
    
    nonisolated func loadingCompleted(models: LoadedModels) {
        print("✅ Model loading completed successfully")
        
        // Switch to the main actor for UI updates
        Task { @MainActor in
            // Use our helper to update the UI with explicit 100%
            self.updateUI(
                progress: 1.0,
                status: "Model Loaded (100%)",
                isLoading: false,
                isLoaded: true
            )
            
            // Clear error flag
            self.hasLoadingError = false
            
            // Extra safety check to ensure model loaded state is set
            self.isModelLoaded = true
            self.loadingProgress = 1.0
            self.loadingProgressString = "100%"
            
            // Force UI update again
            self.objectWillChange.send()
            
            // Force UI to refresh with multiple techniques to ensure UI shows 100%
            self.forceUIRefresh()
            
            // Delay and force it again to ensure the UI updates
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                self.forceUIRefresh()
                print("🔄 DELAYED FORCE UI REFRESH: progress=\(self.loadingProgressString), isLoaded=\(self.isModelLoaded)")
            }
            
            if let currentModelId = self.currentModelId {
                // Post notification that model loading finished
                NotificationCenter.default.post(
                    name: Notification.Name("ModelLoadingFinished"),
                    object: currentModelId
                )
                
                // Post notification that model loading is complete
                NotificationCenter.default.post(
                    name: Notification.Name("ModelLoadingCompleted"),
                    object: currentModelId
                )
            }
            
            // Clear the model loader reference
            self.currentModelLoader = nil
        }
    }
    
    nonisolated func loadingFailed(error: Error) {
        print("❌ IS.1: Error loading model: \(error)")
        
        // Switch to the main actor for UI updates
        Task { @MainActor in
            // Format the error message
            let errorMessage = formatErrorMessage(error)
            
            // Add the error to our list of loading errors
            self.loadingErrors.append(errorMessage)
            
            // Use our helper to update the UI
            self.updateUI(
                progress: 0,
                status: "Error: \(errorMessage)",
                isLoading: false,
                isLoaded: false
            )
            
            // Set error flag
            self.hasLoadingError = true
            
            // Post notification that model loading failed, but only if not suppressed
            if !self.suppressInterruptionNotification, let modelId = self.currentModelId {
                NotificationCenter.default.post(
                    name: Notification.Name("ModelLoadingFailed"),
                    object: modelId,
                    userInfo: ["error": errorMessage]
                )
                print("📣 Posted ModelLoadingFailed notification")
                
                // Store the error message
                self.lastLoadingError = errorMessage
            } else {
                print("🔕 Suppressed ModelLoadingFailed notification due to interruption flag")
            }
            
            // Clear model loader on error
            self.currentModelLoader = nil
        }
    }
    
    /// Verifies that all required model files exist
    /// - Parameters:
    ///   - modelPrefix: The prefix of the model (e.g., "llama")
    ///   - numChunks: The number of chunks in the model
    ///   - lutFFN: The LUT value for FFN
    ///   - lutLMHead: The LUT value for LM Head
    ///   - lutEmbeddings: The LUT value for Embeddings
    ///   - modelDir: The URL of the model directory
    /// - Returns: A tuple with a boolean indicating success and a string message
    private func verifyModelFiles(modelPrefix: String, numChunks: Int, lutFFN: Int, lutLMHead: Int, lutEmbeddings: Int?, modelDir: URL) -> (success: Bool, message: String) {
        print("📋 Verifying model files using ONLY the configuration specified in meta.yaml:")
        print("  - Model prefix: \(modelPrefix)")
        print("  - Number of chunks: \(numChunks)")
        print("  - LUT FFN value: \(lutFFN)")
        print("  - LUT LM Head value: \(lutLMHead)")
        print("  - LUT Embeddings value: \(lutEmbeddings != nil ? String(lutEmbeddings!) : "nil")")
        print("  - Model directory: \(modelDir.path)")
        
        let fileManager = FileManager.default
        var missingFiles: [String] = []
        
        // Create a list of all required files
        var requiredFiles: [String] = [
            // Main model components
            lutEmbeddings != nil && lutEmbeddings! > 0 ? "\(modelPrefix)_embeddings_lut\(lutEmbeddings!).mlmodelc" : "\(modelPrefix)_embeddings.mlmodelc",
            lutLMHead > 0 ? "\(modelPrefix)_lm_head_lut\(lutLMHead).mlmodelc" : "\(modelPrefix)_lm_head.mlmodelc",
            "meta.yaml",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        // Add chunk files - this is the only correct format for FFN files
        for i in 1...numChunks {
            let chunkName: String
            if lutFFN > 0 {
                chunkName = String(format: "\(modelPrefix)_FFN_PF_lut\(lutFFN)_chunk_%02dof%02d.mlmodelc", i, numChunks)
            } else {
                chunkName = String(format: "\(modelPrefix)_FFN_PF_chunk_%02dof%02d.mlmodelc", i, numChunks)
            }
            requiredFiles.append(chunkName)
        }
        
        // Define required files within each .mlmodelc directory
        let mlmodelcRequiredFiles = [
            "model.mil",
            "metadata.json",
            "analytics/coremldata.bin",
            "weights/weight.bin",
            "coremldata.bin"
        ]
        
        // First check main files
        print("Total required: \(requiredFiles.count + (requiredFiles.filter { $0.hasSuffix(".mlmodelc") }.count * mlmodelcRequiredFiles.count)) (Directories: \(requiredFiles.filter { $0.hasSuffix(".mlmodelc") }.count), Files: \(requiredFiles.count + (requiredFiles.filter { $0.hasSuffix(".mlmodelc") }.count * mlmodelcRequiredFiles.count) - requiredFiles.filter { $0.hasSuffix(".mlmodelc") }.count))")
        print("Required files count: \(requiredFiles.count + (requiredFiles.filter { $0.hasSuffix(".mlmodelc") }.count * mlmodelcRequiredFiles.count))")
        
        for file in requiredFiles {
            let filePath = modelDir.appendingPathComponent(file).path
            let exists = fileManager.fileExists(atPath: filePath)
            print("Checking file: \(file) - exists: \(exists)")
            
            if !exists {
                missingFiles.append(file)
            }
            
            // For .mlmodelc directories, check their internal files
            if exists && file.hasSuffix(".mlmodelc") {
                for innerFile in mlmodelcRequiredFiles {
                    let innerFilePath = modelDir.appendingPathComponent(file).appendingPathComponent(innerFile).path
                    let innerExists = fileManager.fileExists(atPath: innerFilePath)
                    print("Checking file: \(file)/\(innerFile) - exists: \(innerExists)")
                    
                    if !innerExists {
                        missingFiles.append("\(file)/\(innerFile)")
                    }
                }
            }
        }
        
        if missingFiles.isEmpty {
            print("✅ Model verification successful - all files present")
            return (true, "Model verification successful - all files present")
        } else {
            print("❌ Model verification failed - missing files:")
            for file in missingFiles.prefix(5) {
                print("  - \(file)")
            }
            if missingFiles.count > 5 {
                print("  - ... and \(missingFiles.count - 5) more files")
            }
            let message = "Model verification failed - missing \(missingFiles.count) files"
            return (false, message)
        }
    }
    
    /**
     Calculates the loading progress based on the current phase and progress within that phase.
     This provides a more gradual and accurate progress indication during model loading.
     
     The loading process is divided into three phases:
     - Components (0% - 70%): Loading individual model components (embeddings, lmhead, chunks)
     - Model (70% - 90%): Creating the loaded models object from components
     - Initialization (90% - 100%): Setting up the inference manager
     
     @param phase The current loading phase ("components", "model", or "initialization")
     @param current The current progress within the phase
     @param total The total number of steps in the phase
     @return A Double between 0.0 and 1.0 representing overall loading progress
     */
    private func calculateProgress(phase: String, current: Int, total: Int) -> Double {
        guard total > 0 else { 
            print("⚠️ PROGRESS CALCULATION WARNING: Total is zero or negative, returning 0.0")
            return 0.0 
        }
        
        let progressInPhase = Double(current) / Double(total)
        print("📊 PROGRESS CALCULATION [Raw]: Phase: \(phase), Current: \(current), Total: \(total), Raw Ratio: \(progressInPhase)")
        
        // Determine range for each phase
        switch phase {
        case "components":
            // Component loading: 0% - 70%
            let result = 0.0 + progressInPhase * 0.7
            print("📊 PROGRESS CALCULATION [Components Phase]: \(current)/\(total) = \(progressInPhase) → Applying scaling factor 0.7 → Final result: \(result) (\(Int(result * 100))%)")
            return result
            
        case "model":
            // Model creation: 70% - 90%
            let result = 0.7 + progressInPhase * 0.2
            print("📊 PROGRESS CALCULATION [Model Phase]: \(current)/\(total) = \(progressInPhase) → Applying scaling factor 0.2 and adding 0.7 baseline → Final result: \(result) (\(Int(result * 100))%)")
            return result
            
        case "initialization":
            // Inference initialization: 90% - 100%
            let result = 0.9 + progressInPhase * 0.1
            print("📊 PROGRESS CALCULATION [Initialization Phase]: \(current)/\(total) = \(progressInPhase) → Applying scaling factor 0.1 and adding 0.9 baseline → Final result: \(result) (\(Int(result * 100))%)")
            return result
            
        default:
            // Fallback to linear progress if phase is unknown
            print("📊 PROGRESS CALCULATION [Unknown Phase]: \(current)/\(total) = \(progressInPhase) → Using raw value without scaling → Final result: \(progressInPhase) (\(Int(progressInPhase * 100))%)")
            return progressInPhase
        }
    }
    
    private init() {}  // Private to enforce singleton
    
    /// Cancels the current model loading process
    func cancelModelLoading(reason: CancellationReason = .userInitiated) {
        print("🛑 CANCEL: Cancelling model loading with reason: \(reason)")
            isCancelled = true
        
        // Since lastCancellationReason is @MainActor isolated, we can safely set it
        // from this method which is also on the main actor
            lastCancellationReason = reason
        print("🛑 CANCEL: Set lastCancellationReason to \(reason)")
        
        // Update the cancellation reason for delegate methods (accessible from nonisolated contexts)
            _cancellationReasonForDelegate = reason
            
        // Cancel our task wrapper
        print("🛑 CANCEL: Cancelling modelLoadingTask: \(modelLoadingTask != nil ? "task exists" : "no task")")
            modelLoadingTask?.cancel()
        
        // Call AnemllCore's cancellation method if we have a model loader
        if let modelLoader = currentModelLoader {
            print("🛑 CANCEL: Found currentModelLoader, calling cancelLoading()")
            Task {
                print("🛑 CANCEL: Inside Task, calling modelLoader.cancelLoading()")
                await modelLoader.cancelLoading()
                print("🛑 CANCEL: modelLoader.cancelLoading() completed")
                // Ensure we clear the reference after cancellation
                DispatchQueue.main.async { [weak self] in
                    print("🛑 CANCEL: Clearing currentModelLoader reference")
                    self?.currentModelLoader = nil
                }
            }
        } else {
            // If we don't have a modelLoader, it's important to clear it here too for consistency
            print("🛑 CANCEL: No currentModelLoader found, skipping cancelLoading() call")
            currentModelLoader = nil
        }
        
        // Use the updateUI method for consistent UI updates
        print("🛑 CANCEL: Updating UI to show cancellation")
            updateUI(
                progress: 0.0,
            status: "Model loading cancelled",
                isLoading: false,
                isLoaded: false
            )
            
            // Reset resources
        print("🛑 CANCEL: Resetting resources")
        
        unloadInferenceManager()
        
            self.currentModelId = nil
            self.inferenceManager = nil
            self.tokenizer = nil
        
        // Reset component tracking
            self.totalComponents = 0
            self.loadedComponents = 0
        
        // Ensure hasLoadingError is set appropriately
        if reason == .userInitiated {
            self.hasLoadingError = true
        }
        
        // Reset cancellation flag for future loads
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.isCancelled = false
        }
        
        print("Model loading cancelled and state reset")
    }
    
    /// Checks if model loading can be cancelled
    var canCancelModelLoading: Bool {
        return isLoadingModel && !isModelLoaded
    }
    
    /// Debug method to print the current state of the model loader
    func debugModelLoaderState() {
        print("🔍 DEBUG MODEL LOADER STATE:")
        print("  - isLoadingModel: \(isLoadingModel)")
        print("  - isModelLoaded: \(isModelLoaded)")
        print("  - currentModelId: \(currentModelId ?? "nil")")
        print("  - isCancelled flag: \(isCancelled)")
        print("  - lastCancellationReason: \(lastCancellationReason)")
        print("  - modelLoadingTask exists: \(modelLoadingTask != nil)")
        print("  - currentModelLoader exists: \(currentModelLoader != nil)")
        print("  - loadingProgress: \(loadingProgress)")
        print("  - loadingStatus: \(loadingStatus)")
    }
    
    private func checkCancellation() throws {
        if Task.isCancelled || isCancelled {
            print("Model loading cancelled - stopping current operation")
            throw CancellationError()
        }
    }

    private func loadModelComponent<T>(_ operation: () throws -> T, componentName: String) throws -> T {
        try checkCancellation()
        print("Loading \(componentName)...")
        let result = try operation()
        print("✓ \(componentName) loaded")
        return result
    }
    
    /// Loads a model from a given URL with its YAML configuration
    func loadModel(modelId: String, from url: URL) async throws {
        // First, unload any existing model to free up memory
        print("🧹 Unloading previous model before loading new one")
                unloadModel()
        
        // Continue with the existing implementation...
        // Create a class with an isolated property for thread safety
        actor RetryStateManager {
            var isRetryAttempted = false
            
            func markRetryAttempted() {
                isRetryAttempted = true
            }
            
            func isRetryAttemptedValue() -> Bool {
                return isRetryAttempted
            }
        }
        
        // Create an actor-isolated state manager
        let retryManager = RetryStateManager()
        
        // Use a simple while loop without labels
        while true {
            // Get the current retry state
            let isMLConfigRetryAttempted = await retryManager.isRetryAttemptedValue()
            
            // Cancel any existing loading task with the appropriate reason
            cancelModelLoading(reason: .startingNewModel)
            
            // Reset cancellation flag to ensure we start fresh
            isCancelled = false
            hasLoadingError = false // Reset error state at the start of loading
            
            // Wait a small moment to ensure any previous tasks have fully cancelled
            try? await Task.sleep(nanoseconds: 200_000_000) // 200ms
            
            // Set the current model ID at the beginning so UI can track it
            self.currentModelId = modelId
            
            // Make sure we reset progress to 0% at the very start and update UI state
            updateUI(
                progress: 0.0, 
                status: isMLConfigRetryAttempted ? "Retrying after CoreML error..." : "Checking model configuration...",
                isLoading: true,
                isLoaded: false
            )
            
            let retryStatus = isMLConfigRetryAttempted ? " (retry after MLModelConfiguration error)" : ""
            print("🚀 Starting model loading process for: \(modelId)\(retryStatus) (0%)")
            print("📊 DEBUG: Initial state: isLoadingModel=\(self.isLoadingModel), loadingProgress=\(self.loadingProgress), currentModelId=\(String(describing: self.currentModelId))")
            
            // Create a new task for loading the model
            // Store the task so it can be cancelled if needed
            modelLoadingTask = Task { [weak self] in
                guard let self = self else { return }
                
                do {
                    // Get retry state - use Task for async call
                    let isRetryAttempted = await retryManager.isRetryAttemptedValue()
                    
                    // Publish loading started notification - only for the first attempt
                    if !isRetryAttempted {
                        NotificationCenter.default.post(
                            name: Notification.Name("ModelLoadingStarted"),
                            object: modelId
                        )
                    } else {
                        print("⚠️ This is a retry attempt after MLModelConfiguration error")
                    }
                    
                    // If this is a retry attempt, clean up any CoreML cache directories
                    if await retryManager.isRetryAttemptedValue() {
                        print("🧹 Cleaning CoreML cache directories before retry...")
                        
                        // Clean caches directory first
                        if let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
                            let fileManager = FileManager.default
                            if let cacheContents = try? fileManager.contentsOfDirectory(at: cacheDir, includingPropertiesForKeys: nil) {
                                for item in cacheContents {
                                    if item.lastPathComponent.contains("com.apple.CoreML") || 
                                       item.lastPathComponent.contains("mlmodel") {
                                        print("🗑️ Clearing CoreML cache: \(item.path)")
                                        try? fileManager.removeItem(at: item)
                                    }
                                }
                            }
                        }
                        
                        // Clean Library directory which might also have CoreML caches
                        if let libraryDir = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first {
                            let fileManager = FileManager.default
                            let coreMLDirURL = libraryDir.appendingPathComponent("Caches/com.apple.CoreML")
                            if FileManager.default.fileExists(atPath: coreMLDirURL.path) {
                                print("🗑️ Clearing CoreML cache in Library: \(coreMLDirURL.path)")
                                try? fileManager.removeItem(at: coreMLDirURL)
                            }
                        }
                    }
                    
                    try checkCancellation()
                    
                    var modelPrefix = ""
                    var lutEmbeddings = "none"
                    var lutFFN: Int? = nil
                    var lutLMHead: Int? = nil
                    var numChunks = 2 // Default to 2 chunks if we can't determine
                    
                    // Try to determine modelPrefix and LUT values from meta.yaml first
                    let metaYamlPath = url.appendingPathComponent("meta.yaml")
                    
                    if FileManager.default.fileExists(atPath: metaYamlPath.path) {
                        do {
                            let yamlString = try String(contentsOf: metaYamlPath, encoding: .utf8)
                            
                            // Debug: Show a preview of the yaml file
                            print("\n📋 Meta.yaml preview (first 200 characters):")
                            print(yamlString.prefix(200))
                            print("...\n")
                            
                            // Try to extract model prefix from meta.yaml model_info section
                            if let prefixRange = yamlString.range(of: "model_prefix:\\s*[a-zA-Z0-9_]+", options: .regularExpression) {
                                let prefixString = String(yamlString[prefixRange])
                                if let prefixValue = prefixString.components(separatedBy: ": ").last?.trimmingCharacters(in: .whitespacesAndNewlines) {
                                    modelPrefix = prefixValue
                                    print("Found model_prefix in meta.yaml: \(modelPrefix)")
                                }
                            }
                            
                            // Try to extract LUT values from meta.yaml
                            // We ONLY use the exact values specified in meta.yaml and don't try random combinations
                            if let lutEmbRange = yamlString.range(of: "lut_embeddings:[\\s\\t]*(\\d+|none|None|NONE)", options: .regularExpression) {
                                let lutString = String(yamlString[lutEmbRange])
                                let rawValue = lutString.components(separatedBy: ":").last?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "none"
                                
                                // Check if the value is a number or "none"
                                if rawValue.lowercased() == "none" {
                                    lutEmbeddings = "none"
                                } else if let intValue = Int(rawValue) {
                                    // Store numeric value as string
                                    lutEmbeddings = "\(intValue)"
                                } else {
                                    // Default to "none" for unrecognized values
                                    lutEmbeddings = "none"
                                }
                                print("Found lut_embeddings in meta.yaml: \(lutEmbeddings) (raw value: \(rawValue))")
                            } else {
                                print("No lut_embeddings found in meta.yaml, using default: none")
                            }
                            
                            if let lutFFNRange = yamlString.range(of: "lut_ffn:\\s*\\d+", options: .regularExpression) {
                                let lutString = String(yamlString[lutFFNRange])
                                if let bits = Int(lutString.components(separatedBy: ": ").last?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "") {
                                    lutFFN = bits
                                    print("Found lut_ffn in meta.yaml: \(lutFFN ?? 0)")
                                }
                            } else {
                                print("⚠️ No valid lut_ffn found in meta.yaml. Using only the specified configuration.")
                            }
                            
                            if let lutLMHeadRange = yamlString.range(of: "lut_lmhead:\\s*\\d+", options: .regularExpression) {
                                let lutString = String(yamlString[lutLMHeadRange])
                                if let bits = Int(lutString.components(separatedBy: ": ").last?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "") {
                                    lutLMHead = bits
                                    print("Found lut_lmhead in meta.yaml: \(lutLMHead ?? 0)")
                                }
                            } else {
                                print("⚠️ No valid lut_lmhead found in meta.yaml. Using only the specified configuration.")
                            }
                            
                            // Extract num_chunks
                            if let numChunksRange = yamlString.range(of: "num_chunks:\\s*\\d+", options: .regularExpression) {
                                let numChunksString = String(yamlString[numChunksRange])
                                if let numChunksValue = numChunksString.components(separatedBy: ": ").last?.trimmingCharacters(in: .whitespacesAndNewlines),
                                   let parsedNumChunks = Int(numChunksValue) {
                                    numChunks = parsedNumChunks
                                    print("Found num_chunks in meta.yaml: \(numChunks)")
                                }
                            }
                        } catch {
                            print("Error reading meta.yaml: \(error)")
                        }
                    }
                    
                    // If no prefix found in meta.yaml, try to determine from modelId
                    if modelPrefix.isEmpty {
                        if let firstComponent = modelId.components(separatedBy: "/").first {
                            modelPrefix = firstComponent
                            print("Using modelPrefix from modelId: \(modelPrefix)")
                        }
                    }
                    
                    // If still no prefix, check model directory for files
                    if modelPrefix.isEmpty {
                        let fileManager = FileManager.default
                        if let files = try? fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil) {
                            for file in files {
                                if file.lastPathComponent.hasSuffix("_embeddings.mlmodelc") {
                                    if let prefix = file.lastPathComponent.components(separatedBy: "_embeddings").first {
                                        modelPrefix = prefix
                                        print("Updated modelPrefix from files: \(modelPrefix)")
                                        break
                                    }
                                }
                            }
                        }
                    }
                    
                    // If still no prefix found, use default
                    if modelPrefix.isEmpty {
                        modelPrefix = "deepseek"
                        print("Using default modelPrefix: \(modelPrefix)")
                    }
                    
                    // Update progress
                    self.loadingProgress = 0.1
                    self.loadingStatus = "Verifying model files..."
                    
                    // Perform comprehensive model verification
                    let verification = verifyModelFiles(
                        modelPrefix: modelPrefix,
                        numChunks: numChunks,
                        lutFFN: lutFFN ?? 0,
                        lutLMHead: lutLMHead ?? 0,
                        lutEmbeddings: lutEmbeddings == "none" ? nil : Int(lutEmbeddings),
                        modelDir: url
                    )
                    
                    guard verification.success else {
                        throw InferenceError.inferenceError("Model verification failed: \(verification.message)")
                    }
                    
                    // Update progress
                    self.loadingProgress = 0.2
                    self.loadingStatus = "Checking model directory"
                    
                    // Ensure the directory exists
                    guard FileManager.default.fileExists(atPath: url.path) else {
                        let error = InferenceError.modelPathNotFound
                        self.loadingStatus = "Model directory not found: \(url.path)"
                        self.loadingProgress = 0
                        self.isLoadingModel = false
                        throw error
                    }
                    
                    try checkCancellation()
                    
                    // Look for meta.yaml file
                    let configURL = url.appendingPathComponent("meta.yaml")
                    guard FileManager.default.fileExists(atPath: configURL.path) else {
                        print("Configuration file not found: \(configURL.path)")
                        throw InferenceError.invalidConfig
                    }
                    
                    // Load YAML configuration
                    self.loadingStatus = "Loading model configuration..."
                    self.loadingProgress = 0.25
                    
                    try checkCancellation()
                    
                    do {
                        let config = try YAMLConfig.load(from: configURL.path)
                        
                        // Define embedDirName here so it's available in both scopes
                        let embedDirName = lutEmbeddings == "none" ? 
                            "\(modelPrefix)_embeddings.mlmodelc" :
                            "\(modelPrefix)_embeddings_lut\(Int(lutEmbeddings) ?? 0).mlmodelc"
                        let embedPath = url.appendingPathComponent(embedDirName).path
                        
                        // Sanity check: Log a warning if the embedDirName doesn't exist but an alternative does
                        if !FileManager.default.fileExists(atPath: embedPath) {
                            // Simply log that the expected path doesn't exist
                            print("⚠️ WARNING: Configured embedDirName '\(embedDirName)' doesn't exist")
                        }
                        
                        // Extra logging to debug path issues
                        print("📁 EMBEDDINGS PATH DEBUG:")
                        print("  - modelPrefix: \(modelPrefix)")
                        print("  - lutEmbeddings: \(lutEmbeddings)")
                        print("  - resultEmbedDirName: \(embedDirName)")
                        print("  - fullEmbedPath: \(embedPath)")
                        print("  - embedDirExists: \(FileManager.default.fileExists(atPath: embedPath))")
                        
                        if !FileManager.default.fileExists(atPath: embedPath) {
                            // No alternative searching, just throw the error
                            throw InferenceError.inferenceError("Embeddings model not found at: \(embedPath)")
                        }
                        if !FileManager.default.fileExists(atPath: url.appendingPathComponent((lutLMHead ?? 0) > 0 ? "\(modelPrefix)_lm_head_lut\(lutLMHead ?? 0).mlmodelc" : "\(modelPrefix)_lm_head.mlmodelc").path) {
                            throw InferenceError.inferenceError("LM Head model not found at: \(url.appendingPathComponent((lutLMHead ?? 0) > 0 ? "\(modelPrefix)_lm_head_lut\(lutLMHead ?? 0).mlmodelc" : "\(modelPrefix)_lm_head.mlmodelc").path)")
                        }
                        
                        // Check for any FFN chunk, not just chunk_01
                        let ffnPath01 = url.appendingPathComponent((lutFFN ?? 0) > 0 ? 
                            "\(modelPrefix)_FFN_PF_lut\(lutFFN ?? 0)_chunk_01of\(String(format: "%02d", numChunks)).mlmodelc" :
                            "\(modelPrefix)_FFN_PF_chunk_01of\(String(format: "%02d", numChunks)).mlmodelc").path
                        let chunkExistsAt01 = FileManager.default.fileExists(atPath: ffnPath01)
                        
                        var ffnPathToUse = ffnPath01
                        
                        // If first chunk missing, look for any other chunk
                        if !chunkExistsAt01 {
                            var foundAnyChunk = false
                            
                            // Check all possible chunks
                            for i in 1...numChunks {
                                let chunkPath = url.appendingPathComponent((lutFFN ?? 0) > 0 ? 
                                    "\(modelPrefix)_FFN_PF_lut\(lutFFN ?? 0)_chunk_\(String(format: "%02d", i))of\(String(format: "%02d", numChunks)).mlmodelc" :
                                    "\(modelPrefix)_FFN_PF_chunk_\(String(format: "%02d", i))of\(String(format: "%02d", numChunks)).mlmodelc").path
                                if FileManager.default.fileExists(atPath: chunkPath) {
                                    foundAnyChunk = true
                                    ffnPathToUse = chunkPath
                                    print("Found alternative chunk: \(chunkPath)")
                                    break
                                }
                            }
                            
                            // Also check non-chunked version
                            let nonChunkedPath = url.appendingPathComponent((lutFFN ?? 0) > 0 ? 
                                "\(modelPrefix)_FFN_PF_lut\(lutFFN ?? 0).mlmodelc" :
                                "\(modelPrefix)_FFN_PF.mlmodelc").path
                            if !foundAnyChunk && FileManager.default.fileExists(atPath: nonChunkedPath) {
                                foundAnyChunk = true
                                ffnPathToUse = nonChunkedPath
                                print("Found non-chunked model: \(nonChunkedPath)")
                            }
                            
                            if !foundAnyChunk {
                                throw InferenceError.inferenceError("No FFN chunk models found at: \(url.path)")
                            }
                        }
                        
                        print("Using model paths:")
                        print("Embed path: \(embedPath)")
                        print("LM Head path: \(url.appendingPathComponent((lutLMHead ?? 0) > 0 ? "\(modelPrefix)_lm_head_lut\(lutLMHead ?? 0).mlmodelc" : "\(modelPrefix)_lm_head.mlmodelc").path)")
                        print("FFN path: \(ffnPathToUse)")
                        
                        // List all files in the model directory for debugging
                        if let files = try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil) {
                            print("Files in model directory:")
                            for file in files {
                                print("- \(file.path)")
                                
                                // If it's a directory, list its contents too
                                if let isDir = try? file.resourceValues(forKeys: [.isDirectoryKey]).isDirectory, isDir {
                                    if let subfiles = try? FileManager.default.contentsOfDirectory(at: file, includingPropertiesForKeys: nil) {
                                        print("  Files in \(file.lastPathComponent):")
                                        for subfile in subfiles {
                                            print("  - \(subfile.lastPathComponent)")
                                            
                                            // If this is a weights directory, check for weight.bin
                                            if subfile.lastPathComponent == "weights" {
                                                if let weightFiles = try? FileManager.default.contentsOfDirectory(at: subfile, includingPropertiesForKeys: nil) {
                                                    print("    Files in weights directory:")
                                                    for weightFile in weightFiles {
                                                        print("    - \(weightFile.lastPathComponent)")
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        try checkCancellation()
                        
                        // Parse meta.yaml to determine the number of chunks
                        // This is now redundant since we already parsed numChunks earlier
                        // var numChunks = 2 // Default to 2 chunks if we can't determine
                        // let metaYamlPath = url.appendingPathComponent("meta.yaml")
                        
                        if FileManager.default.fileExists(atPath: metaYamlPath.path) {
                            // We've already parsed this file above, so we can skip this section
                            // But we'll keep the directory-based detection logic below
                        } else {
                            // If meta.yaml doesn't exist, try to determine from directory names
                            let fileManager = FileManager.default
                            do {
                                let contents = try fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
                                
                                let chunkDirs = contents.filter { 
                                    $0.lastPathComponent.contains((lutFFN ?? 0) > 0 ? 
                                        "\(modelPrefix)_FFN_PF_lut\(lutFFN ?? 0)_chunk_" :
                                        "\(modelPrefix)_FFN_PF_chunk_") && 
                                    $0.lastPathComponent.hasSuffix(".mlmodelc") 
                                }
                                
                                if !chunkDirs.isEmpty {
                                    // Extract the total number of chunks from the first chunk directory name
                                    if let firstChunkDir = chunkDirs.first,
                                       let range = firstChunkDir.lastPathComponent.range(of: #"of(\d+)"#, options: .regularExpression) {
                                        let match = String(firstChunkDir.lastPathComponent[range])
                                        if let totalChunks = Int(match.dropFirst(2)) {
                                            numChunks = totalChunks
                                            print("Detected \(numChunks) chunks from directory names")
                                        }
                                    }
                                }
                            } catch {
                                print("Error examining model directory: \(error)")
                            }
                        }
                        
                        try checkCancellation()
                        
                        // Build the model directories list based on the number of chunks
                        var modelDirs = [URL]()
                        
                        // Use embedDirName from outer scope
                        modelDirs.append(url.appendingPathComponent(embedDirName))
                        
                        // Add all chunks
                        for i in 1...numChunks {
                            let chunkName = (lutFFN ?? 0) > 0 ?
                                String(format: "\(modelPrefix)_FFN_PF_lut\(lutFFN ?? 0)_chunk_%02dof%02d.mlmodelc", i, numChunks) :
                                String(format: "\(modelPrefix)_FFN_PF_chunk_%02dof%02d.mlmodelc", i, numChunks)
                            modelDirs.append(url.appendingPathComponent(chunkName))
                        }
                        
                        // Add lm_head
                        modelDirs.append(url.appendingPathComponent((lutLMHead ?? 0) > 0 ? "\(modelPrefix)_lm_head_lut\(lutLMHead ?? 0).mlmodelc" : "\(modelPrefix)_lm_head.mlmodelc"))
                        
                        try checkCancellation()
                        
                        do {
                            // Use embedDirName from above scope
                            let embedPath = url.appendingPathComponent(embedDirName).path
                            
                            if !FileManager.default.fileExists(atPath: embedPath) {
                                // No alternative searching, just throw the error
                                throw InferenceError.inferenceError("Embeddings model not found at: \(embedPath)")
                            }
                            if !FileManager.default.fileExists(atPath: url.appendingPathComponent((lutLMHead ?? 0) > 0 ? "\(modelPrefix)_lm_head_lut\(lutLMHead ?? 0).mlmodelc" : "\(modelPrefix)_lm_head.mlmodelc").path) {
                                throw InferenceError.inferenceError("LM Head model not found at: \(url.appendingPathComponent((lutLMHead ?? 0) > 0 ? "\(modelPrefix)_lm_head_lut\(lutLMHead ?? 0).mlmodelc" : "\(modelPrefix)_lm_head.mlmodelc").path)")
                            }
                            if !FileManager.default.fileExists(atPath: ffnPathToUse) {
                                throw InferenceError.inferenceError("FFN model not found at: \(ffnPathToUse)")
                            }
                            
                            print("Using model paths:")
                            print("Embed path: \(embedPath)")
                            print("LM Head path: \(url.appendingPathComponent((lutLMHead ?? 0) > 0 ? "\(modelPrefix)_lm_head_lut\(lutLMHead ?? 0).mlmodelc" : "\(modelPrefix)_lm_head.mlmodelc").path)")
                            print("FFN path: \(ffnPathToUse)")
                            
                            // Create a modified config with the correct paths
                            let yamlDict: [String: Any] = [
                                "model_path": ffnPathToUse,
                                "tokenizer_model": config.tokenizerModel,
                                "context_length": config.contextLength,
                                "batch_size": config.batchSize,
                                "embed_path": embedPath,
                                "lmhead_path": url.appendingPathComponent((lutLMHead ?? 0) > 0 ? "\(modelPrefix)_lm_head_lut\(lutLMHead ?? 0).mlmodelc" : "\(modelPrefix)_lm_head.mlmodelc").path,
                                "ffn_path": ffnPathToUse,  // Use the detected path directly, YAMLConfig will handle canonicalization
                                "num_chunks": config.numChunks,
                                // Additional metadata to ensure proper path generation in YAMLConfig
                                "model_prefix": modelPrefix,
                                "lut_ffn": lutFFN ?? 0,
                                "lut_lmhead": lutLMHead ?? 0,
                                "lut_embeddings": lutEmbeddings == "none" ? 0 : Int(lutEmbeddings) ?? 0
                            ]
                            
                            // Enhanced debug logging for the model paths
                            print("🔍 YAML CONFIG PATHS:")
                            print("  - model_path: \(yamlDict["model_path"] as? String ?? "nil")")
                            print("  - embed_path: \(yamlDict["embed_path"] as? String ?? "nil")")
                            print("  - lmhead_path: \(yamlDict["lmhead_path"] as? String ?? "nil")")
                            print("  - ffn_path: \(yamlDict["ffn_path"] as? String ?? "nil")")
                            print("  - num_chunks: \(yamlDict["num_chunks"] as? Int ?? 0)")
                            
                            // Verify these paths exist
                            if let ffnPath = yamlDict["ffn_path"] as? String {
                                print("  - ffn_path exists: \(FileManager.default.fileExists(atPath: ffnPath))")
                                
                                // Check if the FFN path follows the expected chunk pattern
                                if ffnPath.contains("_chunk_01of") {
                                    print("  - ffn_path format: ✅ Contains proper chunk format")
                                } else {
                                    print("  - ffn_path format: ⚠️ Does NOT contain proper chunk format")
                                }
                            }
                            
                            let yamlString = try Yams.dump(object: yamlDict)
                            let modifiedConfig = try YAMLConfig(from: yamlString)
                            
                            print("Using modified config paths:")
                            print("Embed path: \(modifiedConfig.embedPath)")
                            print("LM Head path: \(modifiedConfig.lmheadPath)")
                            print("FFN path: \(modifiedConfig.ffnPath)")
                            
                            // Store the context length from the config for later use
                            self.modelContextLength = config.contextLength
                            print("Using model context length: \(self.modelContextLength)")
                            
                            // Clear the component counter since actual loading is about to begin
                            self.loadedComponents = 0
                            print("📊 DEBUG: Reset component counter: \(self.loadedComponents)/\(self.totalComponents)")
                            
                            // Reset progress again to ensure UI updates
                            self.loadingProgress = 0.0
                            self.loadingStatus = "Starting model loading..."
                            
                            // Small delay to ensure UI updates
                            try await Task.sleep(nanoseconds: 50_000_000) // 50ms delay
                            
                            // Create a new config for the model loader
                            let localConfig = try YAMLConfig(from: yamlString)
                            
                            print("📊 DEBUG: Starting actual model loading - Progress: \(self.loadingProgress)")
                            
                            // Check for meta.yaml to determine if v110 flag should be set
                            var shouldUseV110 = false
                            let metaYamlPath = url.appendingPathComponent("meta.yaml")
                            
                            // Print the model path for debugging
                            print("📂 Model directory path: \(url.path)")
                            print("📄 Meta YAML path: \(metaYamlPath.path)")
                            
                            if FileManager.default.fileExists(atPath: metaYamlPath.path) {
                                do {
                                    let yamlContent = try String(contentsOf: metaYamlPath, encoding: .utf8)
                                    
                                    // Pass the model path to the ModelConfiguration initializer
                                    let modelConfig = try ModelConfiguration(from: yamlContent, modelPath: url.path)
                                    
                                    // You can manually override the v110 flag here if needed
                                    // Uncomment the next line and set to true/false to manually control v110
                                    // modelConfig.shouldUseV110 = true
                                    
                                    shouldUseV110 = modelConfig.shouldUseV110
                                    print("📊 Setting v110 flag to \(shouldUseV110) based on model version \(modelConfig.version)")
                                } catch {
                                    print("⚠️ Error reading meta.yaml for v110 check: \(error). Using default v110=false")
                                    print("📂 Model directory path: \(url.path)")
                                }
                            } else {
                                print("⚠️ meta.yaml not found at \(metaYamlPath.path). Using default v110=false")
                                print("📂 Model directory path: \(url.path)")
                            }
                            
                            do {
                                // Create a ModelLoader with this service as the progress delegate
                                let modelLoader = ModelLoader(progressDelegate: self)
                                self.currentModelLoader = modelLoader
                                
                                // Start actual model loading with progress reporting
                                let models = try await modelLoader.loadModel(from: localConfig)
                                
                                print("📊 DEBUG: ModelLoader.loadModel completed")
                                
                                // Set up our inference manager with the loaded models
                                do {
                                    print("📊 DEBUG: Setting up InferenceManager")
                                    print("📊 DEBUG: Context length: \(localConfig.contextLength)")
                                    print("📊 DEBUG: Batch size: \(localConfig.batchSize)")
                                    print("📊 DEBUG: v110 flag: \(shouldUseV110)")
                                    print("📊 DEBUG: Debug level: \(self.debugLevel)")
                                    
                                    self.inferenceManager = try InferenceManager(
                                        models: models,
                                        contextLength: localConfig.contextLength,
                                        batchSize: localConfig.batchSize,
                                        debugLevel: self.debugLevel,  // Pass debug level to show hidden states
                                        v110: shouldUseV110  // Pass the v110 flag based on model version
                                    )
                                    
                                    print("✅ InferenceManager successfully initialized with v110=\(shouldUseV110)")
                                    
                                    
                                } catch {
                                    print("❌ ERROR initializing InferenceManager: \(error.localizedDescription)")
                                    print("❌ Error type: \(type(of: error))")
                                    throw InferenceError.inferenceError("Failed to initialize InferenceManager: \(error.localizedDescription)")
                                }
                                
                                // Initialize tokenizer
                                do {
                                    print("Initializing tokenizer from: \(localConfig.tokenizerModel)")
                                    let tokenizerPath = URL(fileURLWithPath: localConfig.tokenizerModel)
                                    
                                    // Check if tokenizer model file exists
                                    if !FileManager.default.fileExists(atPath: tokenizerPath.path) {
                                        print("❌ ERROR: Tokenizer model file not found at path: \(tokenizerPath.path)")
                                        throw InferenceError.tokenizationFailed
                                    }
                                    
                                    // Add retry logic for tokenizer initialization
                                    var tokenizerRetryCount = 0
                                    let maxRetries = 3
                                    
                                    while tokenizerRetryCount < maxRetries {
                                        do {
                                            if tokenizerRetryCount > 0 {
                                                print("Retrying tokenizer initialization (attempt \(tokenizerRetryCount + 1)/\(maxRetries))")
                                            }
                                            self.tokenizer = try await Tokenizer(modelPath: tokenizerPath.path, debugLevel: 0)
                                            
                                            if self.tokenizer != nil {
                                                print("✅ Tokenizer successfully initialized")
                                                break
                                            } else {
                                                print("⚠️ WARNING: Tokenizer initialization returned nil on attempt \(tokenizerRetryCount + 1)")
                                                tokenizerRetryCount += 1
                                                if tokenizerRetryCount >= maxRetries {
                                                    throw InferenceError.tokenizationFailed
                                                }
                                                try await Task.sleep(nanoseconds: 500_000_000) // 500ms delay between retries
                                            }
                                        } catch {
                                            print("⚠️ WARNING: Tokenizer initialization error on attempt \(tokenizerRetryCount + 1): \(error.localizedDescription)")
                                            tokenizerRetryCount += 1
                                            if tokenizerRetryCount >= maxRetries {
                                                throw error
                                            }
                                            try await Task.sleep(nanoseconds: 500_000_000) // 500ms delay between retries
                                        }
                                    }
                                    
                                    // Verify tokenizer is initialized
                                    if self.tokenizer == nil {
                                        print("❌ ERROR: Failed to initialize tokenizer after \(maxRetries) attempts")
                                        throw InferenceError.tokenizationFailed
                                    }
                                } catch {
                                    print("❌ ERROR initializing Tokenizer: \(error.localizedDescription)")
                                    print("❌ Error type: \(type(of: error))")
                                    throw InferenceError.tokenizationFailed
                                }
                                
                                // Set the current model ID now that loading is complete
                                self.currentModelId = modelId
                                
                                // Clear the model loader reference
                                self.currentModelLoader = nil
                                
                                
                                // Perform warmup inference if enabled
                                if self.warmupOnLoad {
                                        do {
                                            try await self.performWarmupInference()
                                        } catch {
                                            print("⚠️ Warmup inference failed but continuing: \(error)")
                                        }
                                }
                                
                                // Perform warmup inference if enabled
                                if self.testOnLoad {
                                        do {
                                            try await self.performTestInference()
                                        } catch {
                                            print("⚠️ Warmup inference failed but continuing: \(error)")
                                        }
                                }
                                
                                
                                // Note: Model loading completion is handled in the loadingCompleted delegate method,
                                // which will update isModelLoaded, isLoadingModel, etc.
                            } catch {
                                // Let the error propagate to the outer catch block
                                throw error
                            }
                        } catch {
                            // Check if this is an MLModelConfiguration error and log it (but still propagate)
                            let errorDescription = error.localizedDescription
                            if errorDescription.contains("MLModelConfiguration") && errorDescription.contains("functionName") {
                                print("📊 DEBUG: Caught MLModelConfiguration error in Task: \(errorDescription)")
                                print("📊 DEBUG: This error will be handled in the outer retry loop")
                            }
                            
                            // Let the error propagate to the outer catch block
                            throw error
                        }
                    } catch {
                        if error is CancellationError {
                            print("⛔️ Model loading cancelled, reason: \(self._cancellationReasonForDelegate)")
                            self.loadingStatus = "Loading cancelled"
                            print("📊 DEBUG: Resetting progress to 0 due to cancellation")
                            self.loadingProgress = 0
                            self.isLoadingModel = false
                            self.isModelLoaded = false
                            self.hasLoadingError = true // Set error flag
                            
                            // Only post notification if this was a user-initiated cancellation
                            if self._cancellationReasonForDelegate == .userInitiated {
                                NotificationCenter.default.post(
                                    name: Notification.Name("ModelLoadingInterrupted"),
                                    object: self.currentModelId
                                )
                                print("📣 Posted ModelLoadingInterrupted notification - user initiated")
                            } else {
                                print("🔕 Suppressed ModelLoadingInterrupted notification - starting new model")
                            }
                        } else {
                            print("IS.2: ❌ Error loading model: \(error)")
                            self.loadingStatus = "Error: \(error.localizedDescription)"
                            self.loadingProgress = 0
                            self.isLoadingModel = false
                            self.isModelLoaded = false
                            self.hasLoadingError = true // Set error flag
                            self.currentModelLoader = nil // Clear model loader on error
                            
                            // Post notification that model loading failed, but only if not suppressed
                            if !self.suppressInterruptionNotification {
                                NotificationCenter.default.post(
                                    name: Notification.Name("ModelLoadingFailed"),
                                    object: modelId,
                                    userInfo: ["error": error.localizedDescription]
                                )
                                print("📣 Posted ModelLoadingFailed notification")
                                
                                // Store the error message
                                self.lastLoadingError = error.localizedDescription
                            } else {
                                print("🔕 Suppressed ModelLoadingFailed notification due to interruption flag")
                            }
                            
                            // Only throw non-cancellation errors
                            if !(error is CancellationError) {
                                throw error
                            } else {
                                // For cancellation errors, don't propagate
                                print("🛑 Stopping error propagation for cancellation in outer block")
                            }
                        }
                    }
                } catch {
                    if error is CancellationError {
                        print("⛔️ Model loading cancelled, reason: \(self._cancellationReasonForDelegate)")
                        self.loadingStatus = "Loading cancelled"
                        print("📊 DEBUG: Resetting progress to 0 due to cancellation")
                        self.loadingProgress = 0
                        self.isLoadingModel = false
                        self.isModelLoaded = false
                        self.hasLoadingError = true // Set error flag
                        
                        // Only post notification if this was a user-initiated cancellation
                        if self._cancellationReasonForDelegate == .userInitiated {
                            NotificationCenter.default.post(
                                name: Notification.Name("ModelLoadingInterrupted"),
                                object: self.currentModelId
                            )
                            print("📣 Posted ModelLoadingInterrupted notification - user initiated")
                        } else {
                            print("🔕 Suppressed ModelLoadingInterrupted notification - starting new model")
                        }
                    } else {
                        print("❌ IS.3 Error loading model: \(error)")
                        self.loadingStatus = "Error: \(error.localizedDescription)"
                        self.loadingProgress = 0
                        self.isLoadingModel = false
                        self.isModelLoaded = false
                        self.hasLoadingError = true // Set error flag
                        self.currentModelLoader = nil // Clear model loader on error
                        
                        // Post notification that model loading failed, but only if not suppressed
                        if !self.suppressInterruptionNotification {
                            NotificationCenter.default.post(
                                name: Notification.Name("ModelLoadingFailed"),
                                object: modelId,
                                userInfo: ["error": error.localizedDescription]
                            )
                            print("📣 Posted ModelLoadingFailed notification")
                            
                            // Store the error message
                            self.lastLoadingError = error.localizedDescription
                        } else {
                            print("🔕 Suppressed ModelLoadingFailed notification due to interruption flag")
                        }
                        
                        // Only throw non-cancellation errors
                        if !(error is CancellationError) {
                            throw error
                        } else {
                            // For cancellation errors, don't propagate
                            print("🛑 Stopping error propagation for cancellation in outer block")
                        }
                    }
                }
            }
            
            // Wait for the task to complete or throw an error
            do {
                try await modelLoadingTask!.value
                print("Model loading task completed")
                // If we reach here, loading was successful, break the loop
                break
            } catch {
                print("Model loading task failed: \(error)")
                
                // Check for MLModelConfiguration error at the outer level 
                let errorDescription = error.localizedDescription
                if !isMLConfigRetryAttempted && 
                   (errorDescription.contains("MLModelConfiguration") && errorDescription.contains("functionName")) {
                    
                    print("⚠️ Detected MLModelConfiguration functionName error, will retry once")
                    self.loadingStatus = "Encountered MLModelConfiguration error, will retry..."
                    
                    // Set the retry flag to true - this ensures we only retry once
                    await retryManager.markRetryAttempted()
                    
                    // Sleep for a second before retry
                    try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
                    
                    // Continue to the next iteration of the loop
                    continue
                }
                
                // Check if this is a cancellation error and handle it without propagating
                if error is CancellationError {
                    print("🔄 Outer block handling cancellation error - suppressing further propagation")
                    
                    // Update UI state
                    self.loadingStatus = "Loading cancelled"
                    self.loadingProgress = 0
                    self.isLoadingModel = false
                    self.isModelLoaded = false
                    
                    // Only post notification if this was a user-initiated cancellation
                    if self._cancellationReasonForDelegate == .userInitiated {
                        NotificationCenter.default.post(
                            name: Notification.Name("ModelLoadingInterrupted"),
                            object: self.currentModelId
                        )
                        print("📣 Posted ModelLoadingInterrupted notification - user initiated")
                    } else {
                        print("🔕 Suppressed ModelLoadingInterrupted notification - starting new model")
                    }
                    
                    // Never propagate cancellation errors to higher levels
                    return
                }
                
                // For all other errors, update UI and post notification
                print("IS.4 ❌ Error loading model: \(error)")
                self.loadingStatus = "Error: \(error.localizedDescription)"
                self.loadingProgress = 0
                self.isLoadingModel = false
                self.isModelLoaded = false
                self.hasLoadingError = true // Set error flag
                self.currentModelLoader = nil // Clear model loader on error
                
                // Post notification that model loading failed
                if !self.suppressInterruptionNotification {
                    NotificationCenter.default.post(
                        name: Notification.Name("ModelLoadingFailed"),
                        object: modelId,
                        userInfo: ["error": error.localizedDescription]
                    )
                    print("📣 Posted ModelLoadingFailed notification")
                    
                    // Store the error message
                    self.lastLoadingError = error.localizedDescription
                } else {
                    print("🔕 Suppressed ModelLoadingFailed notification due to interruption flag")
                }
                
                // If we've already retried or it's another error, propagate it
                throw error
            }
        }
    }
    
    /// Unloads the current model to free memory
    func unloadInferenceManager () {
            guard let InferenceManager = inferenceManager else {
                print("No inferenceManager to unload")
            return // or break/continue, depending on the context
            }
            InferenceManager.unload()
        }
    func unloadModel() {
    
            // Cancel any ongoing model loading
        cancelModelLoading()

            print("🛑 unloadModel - Resetting resources")
        unloadInferenceManager()
            self.currentModelId = nil
            self.inferenceManager = nil
            self.tokenizer = nil
            
            // Reset component tracking
            self.totalComponents = 0
            self.loadedComponents = 0
        
            
            // Update published property
            DispatchQueue.main.async {
                self.isModelLoaded = false
                self.loadingProgress = 0.0
                self.loadingStatus = ""
            }
            
            // Ensure hasLoadingError is set appropriately
            // Reset cancellation flag for future loads
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.isCancelled = false
            }
    }
    
    /// Gets the maximum number of tokens for generation based on the model's context length
    private func getMaxTokensForGeneration(multiplier: Int = 1) -> Int {
        // Get the model's context length - typically capped at 75% to leave room for prompt
        // With multiplier to allow for longer generation with window shifting
        let maxTokens = Int(Double(modelContextLength) * 0.75) * multiplier
        
        // Cap at reasonable values (minimum of 512, maximum can be much higher with multiplier)
        return max(512, min(maxTokens, 1024*10)) // More conservative maximum to avoid ANE errors
    }
    
    /// Gets the correct EOS token ID based on the model type
    private func getCorrectEosTokenId(modelId: String? = nil) -> Int {
        // Default to the tokenizer's EOS token ID
        let defaultEosTokenId = tokenizer?.eosTokenId ?? 2 // Fallback to 2 if tokenizer is nil
        
        // If no model ID is provided, use the current model ID
        let modelIdToCheck = modelId ?? currentModelId ?? ""
        
        // Check if this is a DeepSeek model
        if modelIdToCheck.lowercased().contains("deepseek") {
            // For DeepSeek models, we need to check which variant we're dealing with
            if modelIdToCheck.lowercased().contains("r1") || modelIdToCheck.lowercased().contains("distill") {
                // For DeepSeek R1 (distilled from Llama), research suggests 128001 might be the correct EOS token
                print("🔄 Using DeepSeek R1-specific EOS token ID: 128001 (based on research for distilled models)")
                return 128001
            } else {
                // For standard DeepSeek models, use the tokenizer's EOS token
                print("🔄 Using tokenizer's EOS token ID: \(defaultEosTokenId)")
                return defaultEosTokenId
            }
        }
        
        print("Using default EOS token ID from tokenizer: \(defaultEosTokenId)")
        return defaultEosTokenId
    }
    
    /// Returns an async stream yielding chunks of the inference result with performance metrics
    func inferStream(modelId: String, input: String, allowLongGeneration: Bool = false) async throws -> AsyncStream<InferenceResult> {
        guard let inferenceManager = inferenceManager, let tokenizer = tokenizer else {
            print("DEBUG - ERROR: Model not loaded")
            throw InferenceError.modelNotLoaded
        }
        guard modelId == currentModelId else {
            print("DEBUG - ERROR: Model ID mismatch - requested: \(modelId), current: \(currentModelId ?? "none")")
            throw InferenceError.modelNotLoaded  // Ensure correct model is loaded
        }
        
        print("DEBUG - Starting inference with input: \"\(input)\"")
        
        // Create token buffer for long generation tracking
        let tokenBuffer = TokenBuffer(tokenizer: tokenizer)
        
        // Prepare messages for chat context
        let messages: [Tokenizer.ChatMessage] = [.user(input)]
        
        // Tokenize input using the chat template
        let tokens = tokenizer.applyChatTemplate(
            input: messages,
            addGenerationPrompt: true  // Add generation prompt as in CLI
        )
        
        // Determine max tokens based on model's context length and multiplier if allowed
        let maxContextTokens = getMaxTokensForGeneration()
        let maxGenerationTokens = allowLongGeneration ? 
            getMaxTokensForGeneration(multiplier: 4) : maxContextTokens
        
        print("DEBUG - Input tokenized, token count: \(tokens.count), max generation tokens: \(maxGenerationTokens)")
        print("DEBUG - Long generation: \(allowLongGeneration ? "enabled" : "disabled")")
        
        // Get the EOS token directly from the tokenizer
        let eosTokenId = tokenizer.eosTokenId
        print("DEBUG - Using EOS token ID from tokenizer: \(eosTokenId)")
        
        // Create a token printer for streaming output
        let tokenPrinter = await TokenPrinter(tokenizer: tokenizer)
        await tokenPrinter.reset()
        
        // Return an AsyncStream to yield tokens as they are generated
        return AsyncStream { (continuation: AsyncStream<InferenceResult>.Continuation) in
            Task {
                do {
                    // Track start time and token count for performance metrics
            let startTime = Date()
                    var generatedTokenCount = 0
                    
                    print("DEBUG - Starting token generation")
                    // Generate response with token callback
                    let (_, prefillTime, stopReason) = try await inferenceManager.generateResponse(
                        initialTokens: tokens,
                        temperature: defaultTemperature,
                        maxTokens: maxGenerationTokens,
                        eosToken: eosTokenId, // Use the model-specific EOS token ID
                        tokenizer: tokenizer,
                        onToken: { token in
                            // Increment token count
                            generatedTokenCount += 1
                            
                            // Add to buffer for long generations
                            tokenBuffer.addToken(token)
                            
                            // Debug: Check if this is EOS token
                            if token == eosTokenId {
                                print("DEBUG - 🛑 EOS token detected at position \(generatedTokenCount)")
                            }
                            
                            // Enhanced debugging: Check for potential alternative EOS tokens
                            let potentialEosTokens = [2, 100257, 100276, 50256, 0, 524]
                            if potentialEosTokens.contains(token) {
                                print("DEBUG - 🔍 Potential alternative EOS token detected: ID \(token) at position \(generatedTokenCount)")
                                print("       Text representation: \"\(tokenizer.decode(tokens: [token], skipSpecialTokens: false))\"")
                            }
                            
                            // Debug token value and representation every 10 tokens
                            if generatedTokenCount % 20 == 0 || generatedTokenCount < 5 {
                                print("DEBUG - Token #\(generatedTokenCount): ID \(token) = \"\(tokenizer.decode(tokens: [token], skipSpecialTokens: false))\"")
                            }
                            
                            // Create a detached task that doesn't block token generation
                            Task.detached { 
                                await tokenPrinter.addToken(token)
                                
                                // Get text - use token buffer for long generations to ensure full context retention
                                let currentText = allowLongGeneration ? 
                                    tokenBuffer.getText() : 
                                    await tokenPrinter.getBuffer()
                                
                                // Calculate current tokens per second
                                let currentTime = Date().timeIntervalSince(startTime)
                                let currentTPS = currentTime > 0 ? Double(generatedTokenCount) / currentTime : 0
                                
                                // Process current text for DeepSeek models (remove think tags)
                                var processedText = currentText
                                if modelId.lowercased().contains("deepseek") {
                                    // Look for properly closed think tags first
                                    if currentText.contains("<think>") && currentText.contains("</think>") {
                                        print("DEBUG - 🧠 THINK TAGS DETECTED: Processing think tags")
                                        
                                        // Find the last occurrence of </think> to ensure we get the final response
                                        if let lastClosingTagRange = currentText.range(of: "</think>", options: .backwards) {
                                            // Extract content after the last closing think tag
                                            processedText = String(currentText[lastClosingTagRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                                            print("DEBUG - 🧠 Extracted content after </think> tag (\(processedText.count) chars)")
                                            
                                            // Check if the extracted text contains special delimiters
                                            if processedText.contains("://") {
                                                print("DEBUG - ⚠️ Extracted text contains '://' delimiter - potential issue")
                                                
                                                // Try to extract clean content before the delimiter
                                                if let delimiterRange = processedText.range(of: "://") {
                                                    let cleanContent = String(processedText[..<delimiterRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                                                    if !cleanContent.isEmpty {
                                                        print("DEBUG - 🧹 Extracted clean content before '://' delimiter (\(cleanContent.count) chars)")
                                                        processedText = cleanContent
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    // Handle case with only opening think tag (no closing tag)
                                    else if currentText.contains("<think>") && !currentText.contains("</think>") {
                                        print("DEBUG - ⚠️ WARNING: Found unclosed <think> tag")
                                        
                                        // Try to find the actual response content after the thinking
                                        let segments = currentText.components(separatedBy: "://").filter({ !$0.isEmpty })
                                        if let firstSegment = segments.first, !firstSegment.contains("<think>") {
                                            processedText = firstSegment.trimmingCharacters(in: .whitespacesAndNewlines)
                                            print("DEBUG - 🧹 Extracted potential response from incomplete thinking: \(processedText.prefix(30))...")
                                        }
                                    }
                                }
                                
                                // Yield both text and metrics
                                continuation.yield(InferenceResult(
                                    text: processedText,
                                    tokensPerSecond: currentTPS,
                                    tokenCount: generatedTokenCount,
                                    windowShifts: tokenBuffer.getWindowShifts(),
                                    isComplete: false
                                ))
                            }
                        },
                        onWindowShift: {
                            // Record window shifts for tracking
                            tokenBuffer.recordWindowShift()
                            print("DEBUG - Window shifted during generation (Total: \(tokenBuffer.getWindowShifts()))")
                        }
                    )
                    
                    // Calculate inference time and tokens per second
                    let inferenceTime = Date().timeIntervalSince(startTime)
                    let tokensPerSecond = inferenceTime > 0 ? Double(generatedTokenCount) / inferenceTime : 0
                    
                    // Log performance metrics
                    print("DEBUG - Inference complete - Prefill time: \(prefillTime * 1000)ms, Stop reason: \(stopReason)")
                    print("DEBUG - Performance metrics - Inference time: \(String(format: "%.2f", inferenceTime))s, Generated tokens: \(generatedTokenCount), T/s: \(String(format: "%.2f", tokensPerSecond))")
                    if tokenBuffer.getWindowShifts() > 0 {
                        print("DEBUG - Required \(tokenBuffer.getWindowShifts()) window shifts during generation")
                    }
                    
                    // Additional debug for stop reason analysis
                    print("\n=============== INFERENCE RESULT ANALYSIS ===============")
                    print("Stop reason: \(stopReason)")
                    
                    // Log whether EOS was detected - different from stop reason as the model might stop for other reasons
                    let wasStoppedByEOS = (stopReason == "EOS_TOKEN" || stopReason == "eos_token")
                    print("Stopped by EOS token: \(wasStoppedByEOS ? "YES" : "NO")")
                    
                    // Get the last 10 tokens generated (or less if fewer were generated)
                    let lastTokenCount = min(10, generatedTokenCount)
                    if lastTokenCount > 0 {
                        // Instead of re-encoding the final text, we'll analyze the last generated tokens directly
                        print("\nLast \(lastTokenCount) tokens of generated text:")
                        print("Note: Cannot show exact token IDs as encode method is unavailable")
                        
                        // Use appropriate text source
                        let finalBufferText = allowLongGeneration ? tokenBuffer.getText() : await tokenPrinter.getBuffer()
                        print("Final text: \"\(finalBufferText)\"")
                        
                        // Provide information about the EOS token
                        print("\nEOS token information:")
                        print("Configured EOS token ID: \(tokenizer.eosTokenId)")
                        print("Used EOS token ID: \(eosTokenId)")
                        print("EOS token representation: \"\(tokenizer.decode(tokens: [eosTokenId], skipSpecialTokens: false))\"")
                        
                        // Information about potential alternative EOS tokens
                        print("\nPotential alternative EOS tokens for DeepSeek models:")
                        print("Token ID 2: \"\(tokenizer.decode(tokens: [2], skipSpecialTokens: false))\"")
                        print("Token ID 100257: \"\(tokenizer.decode(tokens: [100257], skipSpecialTokens: false))\"")
                    }
                    
                    if !wasStoppedByEOS {
                        print("\nWARNING: Generation did not stop due to EOS token. Check for:")
                        print("- Incorrect EOS token ID: Using \(eosTokenId), but model might expect a different value")
                        print("- Max tokens limit reached (\(maxGenerationTokens))")
                        print("- Model training issue with EOS generation")
                        print("- DeepSeek models often use 2 or 100257 as EOS token - consider checking the tokenizer config")
                    }
                    print("================ INFERENCE RESULT END ================\n")
                    
                    // Finalize the stream - use appropriate source for final text
                    let finalResponse = await self.shouldUseTokenBuffer(allowLongGeneration: allowLongGeneration, windowShifts: tokenBuffer.getWindowShifts()) ? 
                        tokenBuffer.getText() : 
                        await tokenPrinter.stop()
                    print("DEBUG - Final response with thinking content: \"\(finalResponse)\"")
                    
                    // IMPORTANT: Yield the final response as is, with thinking content preserved
                                continuation.yield(InferenceResult(
                        text: finalResponse,
                                    tokensPerSecond: tokensPerSecond,
                        tokenCount: generatedTokenCount,
                        windowShifts: tokenBuffer.getWindowShifts(),
                        isComplete: true
                    ))
                    
                    continuation.finish()
                    
                    // Clear the active task reference since we're done
                    await MainActor.run {
                        if self.inferenceIsCancelled {
                            print("🛑 TOKEN GENERATION COMPLETED AFTER CANCELLATION")
                        } else {
                            print("✅ TOKEN GENERATION COMPLETED NORMALLY")
                        }
                        self.activeInferenceTask = nil
                        self.inferenceIsCancelled = false
                    }
                } catch {
                    // Check if the error is due to cancellation
                    if Task.isCancelled || self.inferenceIsCancelled {
                        print("🛑 TOKEN GENERATION TASK CANCELLED")
                        // End the stream with a cancelled result
                        continuation.yield(InferenceResult(
                            text: tokenBuffer.getText(),
                            tokensPerSecond: 0,
                            tokenCount: 0,
                            windowShifts: 0,
                            isComplete: true,
                            wasCancelled: true
                        ))
                        continuation.finish() // Removed unnecessary try
                        
                        // Reset the task
                        await MainActor.run {
                            self.activeInferenceTask = nil
                            self.inferenceIsCancelled = false
                        }
                        
                        return
                    }
                    
                    print("\n❌ ERROR DURING INFERENCE:")
                    print("📍 Error location: inferStream → generateResponse")
                    print("🔴 Error type: \(String(describing: type(of: error)))")
                    print("📄 Description: \(formatErrorMessage(error))")
                    
                    // Yield an error result instead of trying to throw from the continuation
                    continuation.yield(InferenceResult(
                        text: "Error: \(formatErrorMessage(error))",
                        tokensPerSecond: 0,
                        tokenCount: 0,
                        windowShifts: 0,
                        isComplete: true
                    ))
                    continuation.finish()
                }
            }
        }
    }
    
    /// Performs inference with a system prompt and user input
    func inferWithSystemPrompt(
        modelId: String, 
        systemPrompt: String, 
        userInput: String,
        chatHistory: [Tokenizer.ChatMessage] = [], // Default empty array for backward compatibility
        chatId: String? = nil, // Add optional chat ID parameter
        allowLongGeneration: Bool = true // New parameter to enable long generation
    ) async throws -> AsyncStream<InferenceResult> {
        print("🔍 DEBUG: Starting inferWithSystemPrompt")
        print("🔍 DEBUG: Model ID: \(modelId)")
        print("🔍 DEBUG: Current model ID: \(currentModelId ?? "nil")")
        print("🔍 DEBUG: Tokenizer available: \(tokenizer != nil)")
        print("🔍 DEBUG: Inference manager available: \(inferenceManager != nil)")
        
        guard let inferenceManager = inferenceManager, let tokenizer = tokenizer else {
            print("❌ ERROR: Model not loaded - inferenceManager or tokenizer is nil")
            throw InferenceError.modelNotLoaded
        }
        guard modelId == currentModelId else {
            print("❌ ERROR: Model ID mismatch - requested: \(modelId), current: \(currentModelId ?? "nil")")
            throw InferenceError.modelNotLoaded
        }
        
        // Reset cancellation flag at the start of a new inference
        inferenceIsCancelled = false
        
        // Initialize or update conversation state if chat ID is provided
        if let chatId = chatId {
            if currentState == nil || currentState?.chatId != chatId {
                print("🔄 Initializing state for chat: \(chatId)")
                initializeConversationState(for: chatId)
            }
        }
        
        print("\n=============== TOKEN-BASED APPROACH WITH CONTEXT MANAGEMENT ===============")
        print("DEBUG - inferWithSystemPrompt: Using approach with proper context management")
        
        if let chatId = chatId, let state = currentState {
            print("DEBUG - Chat state: ID=\(chatId), TokenCount=\(state.tokenCount), IsNew=\(state.isNewChat)")
        }
        
        // Create a token buffer for complete response storage, especially for long generations
        let tokenBuffer = TokenBuffer(tokenizer: tokenizer)

        // Create a conversation context from chat history and current input
        var allMessages = chatHistory
        
        //allMessages.debugPrintConversationStructure(prefix:"X1: ")

        
        
        // Ensure we've added the latest user message if it's not already in the history
        let userMessage = Tokenizer.ChatMessage.user(userInput)
        if !allMessages.contains(where: { $0.isUser && $0.content == userInput }) {
            allMessages.append(userMessage)
        }
        
        //allMessages.debugPrintConversationStructure(prefix:"X2:")

        
        // Determine max tokens based on model's context length and whether long generation is allowed
        let maxContextTokens = getMaxTokensForGeneration()
        let maxGenerationTokens = allowLongGeneration ? 
            getMaxTokensForGeneration(multiplier: 4) : maxContextTokens
        
        // Define a buffer for tokens added by the final template (BOS, roles, gen prompt)
        let bufferForTemplateTokens = 50
        // Calculate the maximum tokens the message history itself should occupy *before* templating
        let messageTokenLimit = maxContextTokens - bufferForTemplateTokens
        
        print("DEBUG - Context Limits: maxContext=\(maxContextTokens), messageLimit=\(messageTokenLimit), generationLimit=\(maxGenerationTokens)")
        
        // Prepare conversation messages from history
        var conversationMessages = allMessages
        
        // Add system prompt if present or needed
        let hasSystemPrompt: Bool
        if !systemPrompt.isEmpty {
            print("Adding system prompt to conversation")
            conversationMessages.insert(.system(systemPrompt), at: 0)
            hasSystemPrompt = true
            print("System prompt: \"\(systemPrompt)\"")
        }
        else if thinkingModeEnabled && systemPrompt.isEmpty {
            print("Adding thinking mode system prompt")
            conversationMessages.insert(.system(thinkingSystemPrompt), at: 0)
            hasSystemPrompt = true
                    } else {
            conversationMessages.insert(.system(" "), at: 0)
            hasSystemPrompt = true // Treat empty system prompt as still present for trimming logic
        }
        
        conversationMessages.debugPrintConversationStructure(prefix: "Initial Messages (before trim): ")
        
        // Pre-template Token Estimation and Trimming Loop
        // Estimate tokens WITHOUT the final generation prompt structure
        var estimatedTokens = tokenizer.applyChatTemplate(input: conversationMessages, addGenerationPrompt: false).count
        print("DEBUG - Initial estimated token count (no gen prompt): \(estimatedTokens) / Limit: \(messageTokenLimit)")
        
        // Determine minimum messages to keep (system prompt + last user message)
        let minMessagesToKeep = hasSystemPrompt ? 2 : 1
        
        while estimatedTokens > messageTokenLimit && conversationMessages.count > minMessagesToKeep {
            // Determine which message to remove: the oldest one *after* the system prompt
            let removeIndex = hasSystemPrompt ? 1 : 0
            
            // Ensure the index is valid before attempting removal
            if conversationMessages.indices.contains(removeIndex) {
                let removedMessage = conversationMessages.remove(at: removeIndex)
                print("DEBUG - Trimming: Removed message at index \(removeIndex) (Role: \(removedMessage.role)). \(conversationMessages.count) messages left.")
                
                // Re-estimate token count after removal (still without gen prompt)
                estimatedTokens = tokenizer.applyChatTemplate(input: conversationMessages, addGenerationPrompt: false).count
                print("DEBUG - Estimated token count after trimming: \(estimatedTokens)")
            } else {
                // This shouldn't happen with the count check, but added as safety
                print("DEBUG - Trimming Error: Attempted to remove message at invalid index \(removeIndex). Stopping trim.")
                break
            }
        }
        
        if estimatedTokens > messageTokenLimit {
            print("⚠️ WARNING: Estimated tokens (\(estimatedTokens)) still exceed limit (\(messageTokenLimit)) after trimming. Proceeding, but safety net might be needed.")
        } else {
            print("DEBUG - Pre-template trimming complete. Final estimated count: \(estimatedTokens)")
        }
        
        conversationMessages.debugPrintConversationStructure(prefix: "Trimmed Messages (before sanitize): ")
        
        // 2. Apply template to the simplified conversation
        print("\nApplying chat template to simplified conversation...")
        
        // Sanitize messages to handle thinking tags before applying chat template
        
        // Ensure we're using the proper encoding method from the tokenizer that understands
        // chat formats and special tokens like BOS/EOS
        var currentPrompt = tokenizer.applyChatTemplate(
            input: sanitizeMessagesForChatTemplate(messages: conversationMessages),
            addGenerationPrompt: true  // This will add the assistant prompt for us
        )
        
        // Check if we got a valid result
        if currentPrompt.isEmpty {
            print("⚠️ Warning: Template application returned empty token list, using fallback")
            // Use a fallback approach - manually tokenize without template
            currentPrompt = createFallbackPrompt(messages: conversationMessages, tokenizer: tokenizer)
        } else {
            // Log info about the final prompt that will be sent to the model
            let decodedText = tokenizer.decode(tokens: currentPrompt, skipSpecialTokens: false)
            print("\nDecoded prompt preview:")
            print("\"\(decodedText.prefix(100))...\"")
            print("Final prompt token count: \(currentPrompt.count) / Max Allowed: \(maxContextTokens)")
        }
        
        // Safety Net: Hard Truncation
        // This handles cases where the estimation + buffer wasn't perfect or template added unexpected tokens
        if currentPrompt.count > maxContextTokens {
            print("🚨 SAFETY NET: Final prompt (\(currentPrompt.count) tokens) exceeds max context (\(maxContextTokens)). Applying hard truncation.")
            // Truncate from the *beginning* to preserve the end, which contains the crucial generation prompt structure
            let excessTokens = currentPrompt.count - maxContextTokens
            currentPrompt.removeFirst(excessTokens) 
            print("🚨 SAFETY NET: Truncated prompt to \(currentPrompt.count) tokens.")
        }
        
        // 3. Check token count against context length
        // Unused variable removed to fix compiler warning
        print("Current prompt tokens: \(currentPrompt.count), Max allowed: \(maxContextTokens)")
        print("Generation limit: \(maxGenerationTokens) tokens (Long generation: \(allowLongGeneration ? "enabled" : "disabled"))")
        
        // Update token count in state if we have a chat ID
        if let chatId = chatId {
            // For now, just track the prompt size - in the future we could implement more sophisticated tracking
            if let state = currentState, state.chatId == chatId {
                // Don't update the state yet - we'll do that after generation completes
                print("Will update token count after generation: current=\(state.tokenCount), prompt=\(currentPrompt.count)")
            }
        }
        
        // Debug token IDs to understand special tokens
        print("\n🔢 TOKEN ID DEBUG (first 10):")
        let debugTokenCount = min(10, currentPrompt.count)
        for i in 0..<debugTokenCount {
            let tokenId = currentPrompt[i]
            let tokenText = tokenizer.decode(tokens: [tokenId], skipSpecialTokens: false)
            print("Token #\(i): ID \(tokenId) = \"\(tokenText)\"")
        }
        
        // Decode a preview
        let previewTokens = Array(currentPrompt.prefix(min(500, currentPrompt.count)))
        let decodedText = tokenizer.decode(tokens: previewTokens, skipSpecialTokens: false)
        print("\nDecoded prompt preview (first \(previewTokens.count) tokens):")
        print("\"\(decodedText)\"")
        print("\n---ENC of Decoded prompt preview (first \(previewTokens.count) tokens):")

        
        print("=============== SIMPLIFIED APPROACH END ===============\n")
        
        // 4. Handle token generation through AsyncStream
        return AsyncStream<InferenceResult> { (continuation: AsyncStream<InferenceResult>.Continuation) in
            // Create and store the inference task so it can be cancelled later
            self.activeInferenceTask = Task { @Sendable in
                do {
                    // Track metrics
                    let startTime = Date()
                    var generatedTokenCount = 0
                    var totalWindowShifts = 0
                    var isFirstTokenAfterShift = false
                    var lastShiftTime: Date? = nil
                    
                    // Initialize repetition detector and cancellation flag
                    var repetitionDetector = RepetitionDetector(windowSize: 150, ngramSize: 15, threshold: 3)
                    var isCancellationInProgress = false
                    
                    // Create a token printer for accurate decoding
                    let tokenPrinter = await TokenPrinter(tokenizer: tokenizer)
                    await tokenPrinter.reset()
                    
                    print("\n💡 INFERENCE STARTING - Prefill phase beginning with \(currentPrompt.count) tokens")
                    
                    // Add more debug information before inference
                    print("🔍 DEBUG: Current model ID: \(currentModelId ?? "nil")")
                    print("🔍 DEBUG: inferenceManager type: \(type(of: inferenceManager))")
                    print("🔍 DEBUG: Prompt tokens: \(currentPrompt.count)")
                    print("🔍 DEBUG: Max tokens: \(maxGenerationTokens)")
                    print("🔍 DEBUG: Temperature: \(defaultTemperature)")
                    
                    // Validate the tokenizer before proceeding - this can throw
                    try await tokenPrinter.validateTokenizer()
                    
                    // Start inference using generateResponse with correct parameters
                    let (prefillTokenCount, prefillTime, stopReason) = try await inferenceManager.generateResponse(
                        initialTokens: currentPrompt,
                        temperature: defaultTemperature,
                        maxTokens: maxGenerationTokens,
                        eosToken: tokenizer.eosTokenId,
                        tokenizer: tokenizer,
                        onToken: { [tokenizer, tokenBuffer] token in
                            // Check if cancellation is already in progress
                            if isCancellationInProgress {
                                return
                            }
                            
                            // Check cancellation flag
                            if self.inferenceIsCancelled {
                                print("🛑 TOKEN GENERATION CANCELLED - Stopping token callbacks")
                                return
                            }
                            
                            // Check for repetition
                            if self.useRepetitionDetector && repetitionDetector.addToken(token) {
                                // Set flag to prevent re-entry
                                isCancellationInProgress = true
                                inferenceManager.AbortGeneration(Code: AbortReason.repetitionDetected) // Use repetitionDetected value
                                print("🔄 Repetition detected - stopping generation")
                                
                                Task { @MainActor in
                                    self.inferenceIsCancelled = true
                                    // Get the current text before stopping
                                    let finalText = await self.shouldUseTokenBuffer(allowLongGeneration: allowLongGeneration, windowShifts: tokenBuffer.getWindowShifts()) ? 
                                        tokenBuffer.getText() : 
                                        await tokenPrinter.getBuffer()
                                    
                                    // Yield final result and finish the stream
                                    continuation.yield(InferenceResult(
                                        text: finalText,
                                        tokensPerSecond: generatedTokenCount > 0 ? Double(generatedTokenCount) / Date().timeIntervalSince(startTime) : 0,
                                        tokenCount: generatedTokenCount,
                                        windowShifts: tokenBuffer.getWindowShifts(),
                                        isComplete: true
                                    ))
                                    continuation.finish()
                                }
                                return
                            }
                            
                            // Increment token count
                            generatedTokenCount += 1
                            
                            // Add to buffer for complete storage
                            tokenBuffer.addToken(token)
                            
                            // If this is the first token after a window shift, log it and check for repetition
                            if isFirstTokenAfterShift {
                                print("🔍 FIRST TOKEN AFTER WINDOW SHIFT #\(totalWindowShifts): ID \(token) = \"\(tokenizer.decode(tokens: [token], skipSpecialTokens: false))\"")
                                isFirstTokenAfterShift = false
                                
                                // Calculate time since last shift
                                if let lastTime = lastShiftTime {
                                    let timeSinceShift = Date().timeIntervalSince(lastTime)
                                    print("⏱️ Time elapsed for re-prefill: \(String(format: "%.3f", timeSinceShift))s")
                                }
                                
                                // Reset repetition detector after window shift
                                repetitionDetector = RepetitionDetector(windowSize: 150, ngramSize: 15, threshold: 3)
                            }
                            
                            // Create a detached task to avoid blocking token generation
                            Task.detached {
                                // Check cancellation flag again in the detached task
                                if self.inferenceIsCancelled {
                                    return
                                }
                                
                                // Add token to the printer and get current text
                                await tokenPrinter.addToken(token)
                                
                                // Get text - use token buffer for long generations to ensure full context retention
                                let currentText = await self.shouldUseTokenBuffer(allowLongGeneration: allowLongGeneration, windowShifts: tokenBuffer.getWindowShifts()) ? 
                                    tokenBuffer.getText() : 
                                    await tokenPrinter.getBuffer()
                                
                                // Calculate performance metrics
                                let currentTime = Date().timeIntervalSince(startTime)
                                let tokensPerSecond = currentTime > 0 ? Double(generatedTokenCount) / currentTime : 0
                                
                                // Store tokenizer reference before entering closure
                                let currentTokenizer = await self.tokenizer
                                
                                // Debug token info occasionally
                                if generatedTokenCount % 20 == 0 || generatedTokenCount < 5 {
                                    if let safeTokenizer = currentTokenizer {
                                        print("DEBUG - Token #\(generatedTokenCount): ID \(token) = \"\(safeTokenizer.decode(tokens: [token], skipSpecialTokens: false))\"")
        } else {
                                        print("DEBUG - Token #\(generatedTokenCount): ID \(token) (tokenizer unavailable)")
                                    }
                                    if generatedTokenCount % 200 == 0 && tokenBuffer.getWindowShifts() > 0 {
                                        print("DEBUG - Window shifts so far: \(tokenBuffer.getWindowShifts())")
                                    }
                                }
                                
                                // Final check for cancellation before yielding
                                if self.inferenceIsCancelled {
                                    return
                                }
                                
                                // Yield result to the stream with window shift information
                                continuation.yield(InferenceResult(
                                    text: currentText,
                                    tokensPerSecond: tokensPerSecond,
                                    tokenCount: generatedTokenCount,
                                    windowShifts: tokenBuffer.getWindowShifts(),
                                    isComplete: false // Not the final result
                                ))
                            }
                        },
                        onWindowShift: {
                            // Record window shifts for tracking
                            tokenBuffer.recordWindowShift()
                            totalWindowShifts += 1
                            isFirstTokenAfterShift = true
                            lastShiftTime = Date()
                            
                            // Enhanced window shift logging
                            print("\n🔄 WINDOW SHIFT #\(totalWindowShifts) OCCURRED at \(Date().timeIntervalSince(startTime))s")
                            print("💨 Generated \(generatedTokenCount) tokens before this shift")
                            print("🧠 Current context will be truncated and re-prefilled")
                        }
                    )
                    
                    // Log completion info with enhanced details
                    print("\n✅ GENERATION COMPLETE:")
                    print("📊 Prefill token count: \(prefillTokenCount.count)")
                    // Fix to use the count of the array, not the array itself
                    let prefillTPS = prefillTime > 0 ? Double(prefillTokenCount.count) / prefillTime : 0
                    print("⏱️ Prefill time: \(String(format: "%.3f", prefillTime))s (\(String(format: "%.1f", prefillTPS)) tokens/second)")
                    print("🏁 Generation tokens: \(generatedTokenCount)")
                    print("🔀 Window shifts: \(totalWindowShifts)")
                    print("❓ Stop reason: \(stopReason)")
                    
                    // Update conversation state with the generated tokens
                    if let chatId = chatId {
                        // Update token count with both prompt and generated tokens
                        // This is a simplification - in a real implementation we'd track KV cache state too
                        await MainActor.run {
                            self.updateConversationState(tokenCount: generatedTokenCount, chatId: chatId)
                        }
                    }
                    
                    // Calculate final tokens per second
                    let totalTime = Date().timeIntervalSince(startTime)
                    let tokensPerSecond = totalTime > 0 ? Double(generatedTokenCount) / totalTime : 0
                    
                    // Print performance metrics
                    print("\n📝 PERFORMANCE SUMMARY:")
                    print("⏱️ Total time: \(String(format: "%.2f", totalTime))s")
                    print("🔢 Total tokens: \(generatedTokenCount)")
                    print("⚡ Speed: \(String(format: "%.1f", tokensPerSecond)) tokens/second")
                    print("🧮 Total window shifts: \(totalWindowShifts)")
                    
                    // Final yield with completed text - use token buffer for long generations
                    let finalText = await self.shouldUseTokenBuffer(allowLongGeneration: allowLongGeneration, windowShifts: tokenBuffer.getWindowShifts()) ? 
                        tokenBuffer.getText() : 
                        await tokenPrinter.stop()
                    continuation.yield(InferenceResult(
                        text: finalText,
                        tokensPerSecond: tokensPerSecond,
                        tokenCount: generatedTokenCount,
                        windowShifts: tokenBuffer.getWindowShifts(),
                        isComplete: true // This is the final result
                    ))
                    continuation.finish() // Removed unnecessary try
                    
                    // Clear the active task reference since we're done
                    await MainActor.run {
                        // Store cancellation state before resetting
                        let wasCancelled = self.inferenceIsCancelled
                        
                        // Reset internal state
                        self.activeInferenceTask = nil
                        self.inferenceIsCancelled = false
                        
                        if wasCancelled {
                            print("🛑 TOKEN GENERATION COMPLETED AFTER CANCELLATION")
                        } else {
                            print("✅ TOKEN GENERATION COMPLETED NORMALLY")
                        }
                    }
                } catch {
                    // Check if the error is due to cancellation
                    if Task.isCancelled || self.inferenceIsCancelled {
                        print("🛑 TOKEN GENERATION TASK CANCELLED")
                        // End the stream with a cancelled result
                        continuation.yield(InferenceResult(
                            text: tokenBuffer.getText(),
                            tokensPerSecond: 0,
                            tokenCount: 0,
                            windowShifts: 0,
                            isComplete: true,
                            wasCancelled: true
                        ))
                        continuation.finish() // Removed unnecessary try
                        
                        // Reset the task
                        await MainActor.run {
                            self.activeInferenceTask = nil
                            self.inferenceIsCancelled = false
                        }
                        
                        return
                    }
                    
                    print("\n❌ ERROR DURING INFERENCE:")
                    print("📍 Error location: inferWithSystemPrompt → generateResponse")
                    print("🔴 Error type: \(String(describing: type(of: error)))")
                    print("📄 Description: \(formatErrorMessage(error))")
                    
                    // Yield an error result instead of trying to throw from the continuation
                    continuation.yield(InferenceResult(
                        text: "Error: \(formatErrorMessage(error))",
                        tokensPerSecond: 0,
                        tokenCount: 0,
                        windowShifts: 0,
                        isComplete: true
                    ))
                    continuation.finish()
                }
            }
        }
    }
    
    /// Gets the current model ID
    var currentModel: String? {
        return currentModelId
    }
    
    // MARK: - UI Update Helper
    
    /// Explicitly updates UI properties and ensures they're reflected in the UI
    private func updateUI(progress: Double? = nil, status: String? = nil, isLoading: Bool? = nil, isLoaded: Bool? = nil) {
        // Call this method from main thread if not already there
        if !Thread.isMainThread {
            DispatchQueue.main.async {
                self.updateUI(progress: progress, status: status, isLoading: isLoading, isLoaded: isLoaded)
            }
            return
        }
        
        // We're now on main thread
        
        // First send objectWillChange before any updates
        self.objectWillChange.send()
        
        // Only update properties that are provided
        if let progress = progress {
            self.loadingProgress = progress
            // Also update the string representation which guarantees a change notification
            self.loadingProgressString = "\(Int(progress * 100))%"
            print("📊 UI UPDATE: Progress set to \(self.loadingProgressString)")
        }
        
        if let status = status {
            self.loadingStatus = status
            print("📊 UI UPDATE: Status set to \"\(status)\"")
        }
        
        if let isLoading = isLoading {
            self.isLoadingModel = isLoading
            print("📊 UI UPDATE: isLoadingModel set to \(isLoading)")
        }
        
        if let isLoaded = isLoaded {
            self.isModelLoaded = isLoaded
            print("📊 UI UPDATE: isModelLoaded set to \(isLoaded)")
        }
        
        // Force UI refresh by sending a notification
        NotificationCenter.default.post(name: Notification.Name("ModelLoadingProgressUpdated"), object: nil)
        
        // Send objectWillChange again after updates to ensure SwiftUI views refresh
        self.objectWillChange.send()
    }
    
    /// Specifically updates the loading progress bar to ensure UI reflects changes
    private func updateLoadingProgress(_ percentage: Double) {
        // Call this method from main thread if not already there
        if !Thread.isMainThread {
            DispatchQueue.main.async {
                self.updateLoadingProgress(percentage)
            }
            return
        }
        
        // We're now on main thread
        let intPercentage = Int(percentage * 100)
        print("📊 PROGRESS UPDATE: Setting progress to \(intPercentage)%")
        
        // First send objectWillChange
        self.objectWillChange.send()
        
        // Break the dependency cycle by setting string first, then progress
        // This avoids the progress didSet triggering the string update
        // Store old values temporarily
        let oldProgressString = self.loadingProgressString
        
        // Update string first directly
        self.loadingProgressString = "\(intPercentage)%"
        
        // Then update progress - this will update the string again through didSet,
        // but because we've already set it, there should be no visual change
        self.loadingProgress = percentage
        
        // Extra Debug
        print("📈 PROGRESS CHECK: was \(oldProgressString), now loadingProgress=\(self.loadingProgress), loadingProgressString=\(self.loadingProgressString)")
        
        // Post explicit notification for loading progress update
        NotificationCenter.default.post(
            name: Notification.Name("ExplicitProgressUpdate"),
            object: percentage
        )
        
        // Send objectWillChange again
        self.objectWillChange.send()
    }
    
    // MARK: - Public UI Helper Methods
    
    /// Gets the current loading progress as a percentage string
    public func getProgressString() -> String {
        return loadingProgressString
    }
    
    /// Gets the current loading progress as a Double (0.0 to 1.0)
    public func getProgressValue() -> Double {
        return loadingProgress
    }
    
    /// Gets the current loading status message
    public func getLoadingStatus() -> String {
        return loadingStatus
    }
    
    /// Forces a UI refresh by triggering object changes and notifications
    public func forceUIRefresh() {
        DispatchQueue.main.async {
            // Send multiple change notifications to ensure UI updates
            self.objectWillChange.send()
            
            // Post the full set of notifications
            NotificationCenter.default.post(name: Notification.Name("LoadingProgressChanged"), object: self.loadingProgress)
            NotificationCenter.default.post(name: Notification.Name("ModelLoadedChanged"), object: self.isModelLoaded)
            NotificationCenter.default.post(name: Notification.Name("ModelLoadingProgressUpdated"), object: nil)
            NotificationCenter.default.post(name: Notification.Name("ExplicitProgressUpdate"), object: self.loadingProgress)
            
            // Log the forced refresh
            print("🔄 FORCE UI REFRESH: progress=\(self.loadingProgressString), isLoaded=\(self.isModelLoaded), isLoading=\(self.isLoadingModel)")
            
            // Send another change notification after a tiny delay
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                self.objectWillChange.send()
            }
        }
    }
    
    /// Forces cancellation of model loading with debug information, even if currentModelLoader is nil
    func forceCancel() {
        print("⚠️ FORCE CANCEL: Starting forced cancellation")
        
        // Debug current state
        debugModelLoaderState()
        
        // Set cancellation flag
        isCancelled = true
        inferenceIsCancelled = true  // Set inference cancellation flag
        if let inferenceManager = inferenceManager {
            inferenceManager.AbortGeneration(Code: AbortReason.userCancelled) // Use userCancelled value
        }
        lastCancellationReason = .userInitiated
        _cancellationReasonForDelegate = .userInitiated
        
        // Cancel any active inference task
        print("⚠️ FORCE CANCEL: Cancelling activeInferenceTask")
        activeInferenceTask?.cancel()
        activeInferenceTask = nil
        
        // Cancel task
        print("⚠️ FORCE CANCEL: Cancelling modelLoadingTask")
        modelLoadingTask?.cancel()
        
        // Call directly into loadingCancelled to ensure proper cleanup
        print("⚠️ FORCE CANCEL: Manually calling loadingCancelled delegate method")
        Task {
            loadingCancelled()
        }
        
        // Also trigger normal cancellation in case modelLoader exists
        print("⚠️ FORCE CANCEL: Also calling standard cancelModelLoading")
        cancelModelLoading(reason: .userInitiated)
        
        // Update UI immediately
        updateUI(
            progress: 0.0,
            status: "Model loading cancelled (forced)",
            isLoading: false,
            isLoaded: false
        )
        
        // Reset resources
        print("⚠️ FORCE CANCEL: Resetting all resources")
        unloadInferenceManager()
        self.currentModelId = nil
        self.inferenceManager = nil
        self.tokenizer = nil
        self.totalComponents = 0
        self.loadedComponents = 0
        self.hasLoadingError = true
        
        // Force UI to refresh
        forceUIRefresh()
        
        print("⚠️ FORCE CANCEL: Completed forced cancellation")
    }
    
    /// Cancels only the token generation without unloading the model
    func cancelTokenGeneration() {
        print("🛑 TOKEN GENERATION: Cancelling active inference task")
        
        // Set the inference cancellation flag
        inferenceIsCancelled = true
        
        // Start the cancellation task in background
        Task {
            // Try to cancel the task with retries
            let maxRetries = 20
            var currentRetry = 0
            
            while currentRetry < maxRetries {
                if let task = activeInferenceTask {
                    print("🛑 TOKEN GENERATION: Found active task to cancel")
                    task.cancel()
                    activeInferenceTask = nil
                    print("🛑 TOKEN GENERATION: Cancellation complete")
                    return
                }
                
                // Wait for 100ms before next retry
                try? await Task.sleep(nanoseconds: 100_000_000) // 100ms in nanoseconds
                currentRetry += 1
                print("🛑 TOKEN GENERATION: Waiting for task to become available (attempt \(currentRetry)/\(maxRetries))")
            }
            
            // If we get here, we've exhausted all retries
            if activeInferenceTask != nil {
                print("⚠️ TOKEN GENERATION: Force killing task after \(maxRetries) retries")
                activeInferenceTask = nil
            } else {
                print("🛑 TOKEN GENERATION: No active task found after \(maxRetries) retries")
            }
        }
    }
    
    // MARK: - Conversation State Management
    
    /// Initialize a new conversation state
    func initializeConversationState(for chatId: String) {
            print("🔄 Initializing new conversation state for chat: \(chatId)")
            currentState = ConversationState(chatId: chatId, isNewChat: true)
    }
    
    /// Reset the conversation state for a chat
    func resetConversationState(for chatId: String) {
            print("🔄 Resetting conversation state for chat: \(chatId)")
            if let existingState = currentState, existingState.chatId == chatId {
                currentState = ConversationState(chatId: chatId, isNewChat: false)
            } else {
                initializeConversationState(for: chatId)
        }
    }
    
    /// Update the conversation state after message generation
    func updateConversationState(tokenCount: Int, chatId: String) {
        guard var state = currentState, state.chatId == chatId else {
            print("⚠️ Cannot update state - no active state for chat: \(chatId)")
            initializeConversationState(for: chatId)
            currentState?.tokenCount = tokenCount
                return
            }
            
        state.tokenCount += tokenCount
        state.lastMessageTimestamp = Date()
        state.isNewChat = false
        currentState = state
        
        print("✅ Updated conversation state: \(tokenCount) new tokens, total: \(state.tokenCount)")
    }
    
    /// Check if the conversation needs token trimming
    func needsTokenTrimming(chatId: String) -> Bool {
            guard let state = currentState, state.chatId == chatId else {
                return false
            }
            
            // Get model context length (or use default)
            let maxTokens = modelContextLength
            let tokenLimit = maxTokens - 100 // Leave room for generation
            
            return state.tokenCount > tokenLimit
    }
    
    /// Trims a conversation to fit within context window by replacing older messages with PAD tokens
    /// This preserves the structure of the conversation while reducing token count
    func trimConversationContext(messages: [Tokenizer.ChatMessage], maxTokens: Int) -> [Tokenizer.ChatMessage] {
        guard let tokenizer = tokenizer else {
            return messages // Can't trim without tokenizer
        }
        
        // First check if we actually need to trim
        var tokenCount = 0
        for message in messages {
            let tokens = tokenizer.tokenize(message.content)
            tokenCount += tokens.count
        }
        
        if tokenCount <= maxTokens {
            print("No trimming needed, token count \(tokenCount) is within limit \(maxTokens)")
            return messages
        }
        
        print("Trimming conversation with \(tokenCount) tokens to fit \(maxTokens) limit")
        
        // We need to trim. Strategy: Replace the middle of older messages with PAD tokens
        // Keep the beginning and end of each message for context
        var trimmedMessages: [Tokenizer.ChatMessage] = []
        var remainingBudget = maxTokens
        let padTokenId = tokenizer.padTokenId
        
        // Keep at least the most recent 2 messages untouched
        let recentMessages = min(2, messages.count)
        let recentMessagesBudget = Int(Double(maxTokens) * 0.7) // Allocate 70% to recent messages
        var recentTokenCount = 0
        
        // Count tokens in recent messages (from the end)
        for i in (messages.count - recentMessages)..<messages.count {
            let tokens = tokenizer.tokenize(messages[i].content)
            recentTokenCount += tokens.count
        }
        
        // If recent messages are too large, we need a more aggressive strategy
        if recentTokenCount > recentMessagesBudget {
            print("Recent messages are too large (\(recentTokenCount) tokens), using aggressive trimming")
            // For aggressive trimming, we'd need to trim even recent messages
            // This would be more complex - for now return all messages and let normal truncation happen
            return messages
        }
        
        // Adjust remaining budget for older messages
        remainingBudget -= recentTokenCount
        
        // Process older messages (all except the last 'recentMessages')
        for i in 0..<(messages.count - recentMessages) {
            let message = messages[i]
            let tokens = tokenizer.tokenize(message.content)
            
            if tokens.count <= 10 || remainingBudget >= tokens.count {
                // For very short messages or if we have enough budget, keep them as is
                trimmedMessages.append(message)
                remainingBudget -= tokens.count
            } else {
                // For longer messages, keep the start and end, replace middle with PAD
                let prefixLength = min(5, tokens.count / 3)
                let suffixLength = min(5, tokens.count / 3)
                let prefixTokens = Array(tokens.prefix(prefixLength))
                let suffixTokens = Array(tokens.suffix(suffixLength))
                
                // Create a new message with: prefix + PAD + suffix
                var newTokens = prefixTokens
                // Add a single PAD token as placeholder for trimmed content
                newTokens.append(padTokenId)
                newTokens.append(contentsOf: suffixTokens)
                
                // Create the trimmed content
                let trimmedContent = tokenizer.detokenize(newTokens)
                let trimmedMessage = message.isUser ? 
                    Tokenizer.ChatMessage.user(trimmedContent) :
                    Tokenizer.ChatMessage.assistant(trimmedContent)
                
                trimmedMessages.append(trimmedMessage)
                remainingBudget -= newTokens.count
                
                print("Trimmed message from \(tokens.count) to \(newTokens.count) tokens")
            }
            
            // If we're out of budget, stop processing older messages
            if remainingBudget <= 0 {
                break
            }
        }
        
        // Add all recent messages (untouched)
        for i in (messages.count - recentMessages)..<messages.count {
            trimmedMessages.append(messages[i])
        }
        
        print("Conversation trimmed: \(messages.count) messages → \(trimmedMessages.count) messages")
        return trimmedMessages
    }
    
    /// Creates a simple fallback prompt when the chat template fails
    private func createFallbackPrompt(messages: [Tokenizer.ChatMessage], tokenizer: Tokenizer) -> [Int] {
        print("Using fallback prompt creation method - improved version from chat_full.py")
        
        var prompt = [Int]()
        
        // Check if this is a DeepSeek model (based on tokens we see in the log)
        let isDeepSeek = currentModelId?.lowercased().contains("deepseek") ?? false
        
        // Use model-specific tags
        let userTag = isDeepSeek ? "<|User|>" : "<|user|>"
        let assistantTag = isDeepSeek ? "<|Assistant|>" : "<|assistant|>"
        
        // Add beginning of sequence token first
        if tokenizer.bosTokenId > 0 {
            prompt.append(tokenizer.bosTokenId)
            print("Added BOS token: \(tokenizer.bosTokenId)")
        }
        
        // In chat_full.py style, we format the chat history using explicit tags
        for (_, message) in messages.enumerated() {
            // Add role tag based on message type
            if message.isUser {
                // For user messages, tokenize the appropriate user tag
                let userTagTokens = tokenizer.tokenize(userTag)
                prompt.append(contentsOf: userTagTokens)
                print("Added user tag: \(userTag) - \(userTagTokens.count) tokens")
            } else if message.isAssistant {
                // For assistant messages, tokenize the appropriate assistant tag
                let assistantTagTokens = tokenizer.tokenize(assistantTag)
                prompt.append(contentsOf: assistantTagTokens)
                print("Added assistant tag: \(assistantTag) - \(assistantTagTokens.count) tokens")
            }
            
            // Add message content directly (not through chat template)
            let contentTokens = tokenizer.tokenize(message.content)
            prompt.append(contentsOf: contentTokens)
            
            let contentPreview = message.content.isEmpty ? "(empty)" : 
                "\(message.content.prefix(min(30, message.content.count)))\(message.content.count > 30 ? "..." : "")"
            print("Added message content: \(contentPreview) - \(contentTokens.count) tokens")
        }
        
        // For the final assistant turn, add the assistant tag to prime for generation
        if !messages.isEmpty && !messages.last!.isAssistant {
            let assistantTagTokens = tokenizer.tokenize(assistantTag)
            prompt.append(contentsOf: assistantTagTokens)
            print("Added final assistant tag for generation: \(assistantTag) - \(assistantTagTokens.count) tokens")
        }
        
        print("Created fallback prompt with \(prompt.count) tokens")
        
        // Debug: decode the prompt to see what we've created
        let decodedPrompt = tokenizer.detokenize(prompt)
        print("Fallback prompt preview: \"\(decodedPrompt.prefix(100))...\"")
        
        return prompt
    }
    
    // MARK: - Helper Methods for Batch Inference
    
    /// Pads a sequence of token IDs to a consistent length for batch inference
    /// - Parameters:
    ///   - sequence: The original token sequence
    ///   - targetLength: The desired length after padding
    ///   - rightPad: Whether to pad on the right (true) or left (false)
    /// - Returns: A padded sequence of exactly targetLength
    private func padSequence(_ sequence: [Int], toLength targetLength: Int, rightPad: Bool = true) -> [Int] {
        guard let tokenizer = tokenizer else {
            return sequence // Can't pad without tokenizer
        }
        
        // If sequence is longer than target, truncate it
        if sequence.count > targetLength {
            return rightPad 
                ? Array(sequence.prefix(targetLength))  // Keep beginning of sequence
                : Array(sequence.suffix(targetLength))  // Keep end of sequence
        }
        
        // If sequence is already the right length, return it as is
        if sequence.count == targetLength {
            return sequence
        }
        
        // Create padding array of the right size
        let padToken = tokenizer.padTokenId
        let paddingNeeded = targetLength - sequence.count
        let padding = Array(repeating: padToken, count: paddingNeeded)
        
        // Return padded sequence
        return rightPad
            ? sequence + padding  // Padding at the end
            : padding + sequence  // Padding at the beginning
    }
    
    /// Prepare multiple prompts for batch inference by padding all to the same length
    /// - Parameters:
    ///   - prompts: Array of token sequences to batch together
    ///   - rightPad: Whether to pad on the right (true) or left (false)
    /// - Returns: Padded prompts all of the same length
    private func prepareBatchPrompts(_ prompts: [[Int]], rightPad: Bool = true) -> [[Int]] {
        if prompts.isEmpty {
            return []
        }
        
        // Find the maximum sequence length
        let maxLength = prompts.map { $0.count }.max() ?? 0
        
        // Pad all sequences to the same length
        return prompts.map { sequence in
            padSequence(sequence, toLength: maxLength, rightPad: rightPad)
        }
    }
    
    // MARK: - Thinking Mode Support
    
    /// Thinking mode encourages the model to show its reasoning process inside <think> tags
    private var thinkingModeEnabled: Bool = false
    private let thinkingSystemPrompt = """
    You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
    """
    
    /// Enable or disable thinking mode
    public func setThinkingMode(_ enabled: Bool) {
        thinkingModeEnabled = enabled
        print("Thinking mode \(enabled ? "enabled" : "disabled")")
    }
    
    /// Get the current thinking mode state
    public func isThinkingModeEnabled() -> Bool {
        return thinkingModeEnabled
    }
    
    /// Sanitize messages with thinking tags to ensure chat template works properly
    // This function is used to clean up the messages before they're passed to the chat template, ensuring that any 
    // intermediate "thinking" content is properly handled and only the final responses are included in the conversation.
    
    private func sanitizeMessagesForChatTemplate(messages: [Tokenizer.ChatMessage]) -> [Tokenizer.ChatMessage] {
        return messages.map { message in
            // Only need to process assistant messages
            if message.isAssistant {
                let content = message.content
                
                // Check if this message contains thinking tags
                if content.contains("<think>") {
                    // If it contains complete thinking tags, extract the final response
                    if content.contains("</think>") {
                        // Get everything after the last </think> tag
                        if let range = content.range(of: "</think>", options: .backwards) {
                            let finalResponse = String(content[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                            
                            // If there's actual content after the </think> tag, use it
                            if !finalResponse.isEmpty {
                                print("Sanitized thinking message - extracted final response")
                                return Tokenizer.ChatMessage.assistant(finalResponse)
                            }
                            
                            // Otherwise remove the thinking content entirely
                            let thinkStart = content.range(of: "<think>")!
                            let responseBefore = String(content[..<thinkStart.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                            if !responseBefore.isEmpty {
                                print("Sanitized thinking message - used content before <think>")
                                return Tokenizer.ChatMessage.assistant(responseBefore)
                            }
                            
                            // If still nothing, return a simplified response
                            print("Sanitized thinking message - no usable content found, using empty response")
                            return Tokenizer.ChatMessage.assistant("")
                        }
                    }
                    
                    // For unclosed think tags, just strip them out
                    let sanitized = content.replacingOccurrences(of: "<think>", with: "")
                    print("Sanitized unclosed thinking tags")
                    return Tokenizer.ChatMessage.assistant(sanitized)
                }
            }
            
            // For other messages, return as is
            return message
        }
    }
    
    // Add a helper method to determine if we should use the token buffer for full text
    private func shouldUseTokenBuffer(allowLongGeneration: Bool, windowShifts: Int) async -> Bool {
        // Always use token buffer when we've had window shifts to ensure full text retention
        // Also use it when long generation is enabled for consistency
        return windowShifts > 0 || allowLongGeneration
    }
    
    // Add this debug method for hidden states
    private func debugHiddenStates(_ hiddenStates: MLMultiArray, label: String, debugLevel: Int = 1) {
        if debugLevel > 0 {
            let shape = hiddenStates.shape.map { $0.intValue }
            print("\n🔍 DEBUG HIDDEN STATES: \(label)")
            print("  Shape: \(shape)")
            
            // Print first 8 values
            print("  First 8 values:")
            var firstValues = "  "
            for i in 0..<min(8, hiddenStates.count) {
                let value = Float(truncating: hiddenStates[i])
                firstValues += String(format: "%.6f ", value)
            }
            print(firstValues)
            
            // Print last 8 values
            print("  Last 8 values:")
            var lastValues = "  "
            for i in max(0, hiddenStates.count-8)..<hiddenStates.count {
                let value = Float(truncating: hiddenStates[i])
                lastValues += String(format: "%.6f ", value)
            }
            print(lastValues)
        }
    }
    
    // Add a method to set debug level
    func setDebugLevel(_ level: Int) {
        self.debugLevel = level
        print("Debug level set to \(level) - restart model to apply changes")
    }
    
    @MainActor
    private func startModelLoading(model: Model, modelPath: URL) {
        // Ensure any existing model is unloaded before starting to load a new one
        InferenceService.shared.unloadModel()
        
        // Clear any previous error
        self.lastLoadingError = nil
        
        // Start the loading process with high priority
        Task(priority: .userInitiated) {
            do {
                print("🚀 Starting model loading for: \(model.id) (0%)")
                
                // Set loading state in InferenceService
                InferenceService.shared.loadingStatus = "Starting model loading..."
                InferenceService.shared.isLoadingModel = true
                InferenceService.shared.loadingProgress = 0.05
                
                // Post notification that model loading has started
                NotificationCenter.default.post(
                    name: Notification.Name("ModelLoadingStarted"),
                    object: model.id
                )
                
                // Start loading the model
                try await InferenceService.shared.loadModel(modelId: model.id, from: modelPath)
                
                print("✅ Model loaded successfully: \(model.id) (100%)")
                
                // Clear any previous error on successful load
                self.lastLoadingError = nil
                
            } catch {
                print("❌ MS.1 Error loading model: \(formatErrorMessage(error))")
                
                // Don't post failure notifications for cancellation errors
                if error is CancellationError {
                    print("🔄 Model loading was cancelled - not posting error notification")
                    
                    // Just reset loading state without posting error
                    InferenceService.shared.isLoadingModel = false
                    InferenceService.shared.loadingProgress = 0
                    InferenceService.shared.loadingStatus = ""
                    return
                }
                
                // Reset loading state
                InferenceService.shared.isLoadingModel = false
                InferenceService.shared.loadingProgress = 0
                InferenceService.shared.loadingStatus = "Error: \(formatErrorMessage(error))"
                
                // Store the error message
                self.lastLoadingError = formatErrorMessage(error)
                
                // Post notification that model loading failed (only for non-cancellation errors)
                NotificationCenter.default.post(
                    name: Notification.Name("ModelLoadingFailed"),
                    object: model.id,
                    userInfo: ["error": formatErrorMessage(error)]
                )
            }
        }
    }

    /// Configure whether to perform warmup inference after model loading
    func configureWarmup(enabled: Bool) {
        print("🔧 Configuring warmup inference: \(enabled ? "enabled" : "disabled")")
        warmupOnLoad = enabled
    }
    
    // Add public accessor for inferenceManager
    var debugInferenceManager: InferenceManager? {
        return inferenceManager
    }
    
    // Add a method to format error messages
    private func formatErrorMessage(_ error: Error) -> String {
        if let modelError = error as? AnemllCore.ModelError {
            // Check if the error description contains the error code
            let description = modelError.localizedDescription
            
            if description.contains("`.functionName` property must be `nil`") {
                return "Model loading Error: OS CoreML cache is full. Recommend to reboot your device and retry!"
            }else if description.contains("error 3") {
                return "Model loading cancelled"
            } else if description.contains("error 1") {
                return "Model loading cancelled"
            } else if description.contains("error 2") {
                return "Invalid model path"
            } else if description.contains("error 4") {
                return "Invalid model configuration"
            } else if description.contains("cancelled") || description.contains("cancel") {
                return "Model loading cancelled"
            } else if description.contains("path") {
                return "Invalid model path"
            } else if description.contains("not found") {
                return "Model not found"
            } else if description.contains("configuration") {
                return "Invalid model configuration"
            } else {
                return "Model loading error: \(description)"
            }
        } else if let inferenceError = error as? InferenceError {
            switch inferenceError {
            case .modelNotLoaded:
                return "Model not loaded or not found"
            case .contextTooLong:
                return "Context exceeds maximum length"
            case .tokenizationFailed:
                return "Failed to tokenize input"
            case .inferenceError(let message):
                return message
            case .modelPathNotFound:
                return "Model path not found"
            case .invalidConfig:
                return "Invalid model configuration"
            }
        } else if error is CancellationError {
            return "Model loading cancelled"
        } else {
            return error.localizedDescription
        }
    }
    
}

// Add the extension before the InferenceService class
extension Array where Element == Tokenizer.ChatMessage {
    func debugPrintConversationStructure(prefix: String = "") {
        print("\n\(prefix)DEBUGGING CONVERSATION STRUCTURE:")
        for (index, message) in self.enumerated() {
            let contentPreview = message.content.isEmpty ? "(empty)" : 
            "\(message.content.prefix(Swift.min(100, message.content.count)))\(message.content.count > 30 ? "..." : "")"
            print("\(prefix)Message #\(index): Role: \(message.role), Content: \(contentPreview)")
        }
    }
}
    
