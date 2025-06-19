import Foundation
import ArgumentParser
import AnemllCore
import CoreML
import CoreFoundation
import Dispatch
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

// Update TokenPrinter class to be Sendable-compliant
@globalActor actor TokenPrinterActor {
    static let shared = TokenPrinterActor()
}

@TokenPrinterActor
class TokenPrinter: @unchecked Sendable {
    private let tokenizer: Tokenizer
    private var buffer: String = ""
    private var isThinking: Bool = false  // Start as false
    private var isProcessing: Bool = false
    private let debugLevel: Int
    private let showSpecialTokens: Bool
    
    private var currentTokens: [Int] = []  // Add to track tokens
    
    // Add method to reset state for new message
    func startNewMessage() async {
        buffer = ""
        isThinking = false
        isProcessing = false
        currentTokens = []  // Reset tokens
    }
    
    // Add helper method to detect thinking tokens
    private func isThinkingToken(_ token: Int) -> Bool {
        let withSpecial = tokenizer.decode(tokens: [token], skipSpecialTokens: false)
        return withSpecial == "<think>" || withSpecial == "</think>" ||
               (withSpecial == "think" && buffer.hasSuffix("</"))
    }
    
    init(tokenizer: Tokenizer, debugLevel: Int = 0, showSpecialTokens: Bool = false) {
        self.tokenizer = tokenizer
        self.debugLevel = debugLevel
        self.showSpecialTokens = showSpecialTokens
    }
    
    func addToken(_ token: Int) async {
        isProcessing = true
        currentTokens.append(token)  // Track token
        
        // First decode with special tokens to check for EOS
        let withSpecial = tokenizer.decode(tokens: [token], skipSpecialTokens: false)
        if debugLevel >= 1 {
            print("\nToken \(token): '\(withSpecial)'")
            // Print current window every 10 tokens
            if currentTokens.count % 10 == 0 {
                print("\nCurrent window:")
                print("Tokens:", currentTokens)
                print("Decoded:", tokenizer.decode(tokens: currentTokens))
                print("Window size:", currentTokens.count)
            }
        }
        
        // Check for special tokens and thinking tags
        //let isEOSToken = withSpecial.contains("<|endoftext|>") || withSpecial.contains("</s>")
        //let isEOTToken = withSpecial.contains("<|eot_id|>")
        let isThinkStartToken = withSpecial == "<think>" || 
                               (withSpecial == "<th" && buffer.isEmpty) // Start of <think>
        let isThinkEndToken = withSpecial == "</think>" || 
                             (buffer.hasSuffix("</") && withSpecial == "think")
        
        // Normal decoding for display
        let decoded = tokenizer.decode(tokens: [token])
        let cleanedDecoded = decoded.replacingOccurrences(of: "assistant", with: "")
        
        if isThinkStartToken {
            print("\u{001B}[34m", terminator: "")  // Set blue color at start of <think>
            print(cleanedDecoded, terminator: "")
            if withSpecial == "<th" {
                isThinking = true  // Set thinking mode when we see <th
            }
        } else if isThinkEndToken {
            if withSpecial == "think" {
                print("think>", terminator: "")
            } else {
                print("</think>", terminator: "")
            }
            isThinking = false
            print("\u{001B}[0m", terminator: "")
        } else if isThinking {
            print("\u{001B}[34m\(cleanedDecoded)", terminator: "")
        } else {
            print(cleanedDecoded, terminator: "")
        }
        
        buffer += cleanedDecoded
        fflush(stdout)
        isProcessing = false
    }
    
    func stop() async -> String {
        // Wait for any pending tokens to be processed
        while isProcessing {
            try? await Task.sleep(nanoseconds: 1_000_000)  // 1ms
        }
        
        print("\u{001B}[0m") // Reset color
        fflush(stdout)
        
        // Get final response and clear buffer
        let response = buffer.replacingOccurrences(of: "assistant", with: "")
        buffer = ""  // Clear buffer for next use
        currentTokens = []  // Clear tokens
        return response
    }
    
    func drain() async {
        // Wait for any pending tokens to be processed
        while isProcessing {
            try? await Task.sleep(nanoseconds: 1_000_000)  // 1ms
        }
    }
}

@main
struct AnemllCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "anemllcli",
        abstract: "CLI for running LLM inference with CoreML"
    )
    
    @Option(name: .long, help: "Path to meta.yaml config file")
    var meta: String
    
    @Option(name: .long, help: "Maximum number of tokens to generate")
    var maxTokens: Int? = nil  // Optional parameter
    
    @Option(name: .long, help: "Temperature for sampling (0 for deterministic)")
    var temperature: Float = 0.0
    
    // Add prompt option
    @Option(name: .long, help: "Single prompt to process and exit")
    var prompt: String?
    
    @Option(name: .long, help: "System prompt to use")
    var system: String?
    
    @Option(name: .long, help: "Save assistant's response to file")
    var save: String?
    
    @Option(name: .long, help: "Template style (default, deephermes)")
    var template: String = "deephermes"
    
    @Flag(name: .long, help: "Don't add generation prompt")
    var noGenerationPrompt = false
    
    @Option(name: .long, help: "Debug level (0=disabled, 1=basic, 2=hidden states)")
    var debugLevel: Int = 0
    
    // Add thinking mode flag
    @Flag(name: .long, help: "Enable thinking mode with detailed reasoning")
    var thinkingMode = false
    
    // Add the option to CLI:
    @Flag(name: .long, help: "Show special tokens in output")
    var showSpecialTokens = false
    
    // Add flag for detailed loading progress
    @Flag(name: .long, help: "Show detailed model loading progress")
    var showLoadingProgress = false
    
    // Update thinking prompt to use actual tokens
    private static let THINKING_PROMPT = """
    You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.

    Example:
    <think>
    Let me think about this step by step:
    1. First, I need to understand...
    2. Then, I should consider...
    3. Finally, I can conclude...
    </think>
    Here's my answer...
    """
    
    // ANSI color codes as static constants
    private static let DARK_BLUE = "\u{001B}[34m"
    private static let RESET_COLOR = "\u{001B}[0m"
    
    // MARK: - Progress Delegate Implementation
    // ModelLoadingProgressDelegate implementation for the CLI
    private class CLIProgressDelegate: ModelLoadingProgressDelegate, @unchecked Sendable {
        private let showDetailedProgress: Bool
        private var lastPercentageReported: Int = -1
        private var loadingBar: String = ""
        private var loadingStartTime: CFAbsoluteTime = 0
        
        init(showDetailedProgress: Bool) {
            self.showDetailedProgress = showDetailedProgress
            self.loadingStartTime = CFAbsoluteTimeGetCurrent()
        }
        
        func loadingProgress(percentage: Double, stage: String, detail: String?) {
            let percentInt = Int(percentage * 100)
            
            // Only update if percentage has changed
            if percentInt != lastPercentageReported {
                lastPercentageReported = percentInt
                
                // Basic progress bar (always shown)
                let fullWidth = 30
                let filledWidth = Int(Double(fullWidth) * percentage)
                let emptyWidth = fullWidth - filledWidth
                
                loadingBar = "["
                loadingBar += String(repeating: "■", count: filledWidth)
                loadingBar += String(repeating: "□", count: emptyWidth)
                loadingBar += "] \(percentInt)%"
                
                // Clear the line and reprint
                print("\u{001B}[2K\r\(loadingBar)", terminator: "")
                
                // If detailed progress is enabled, print more info
                if showDetailedProgress && detail != nil {
                    print(" \(stage): \(detail!)")
                } else {
                    fflush(stdout)
                }
            }
        }
        
        func loadingCompleted(models: LoadedModels) {
            let elapsed = CFAbsoluteTimeGetCurrent() - loadingStartTime
            print("\u{001B}[2K\r[" + String(repeating: "■", count: 30) + "] 100%")
            print("✓ Models loaded successfully in \(String(format: "%.2f", elapsed))s")
        }
        
        func loadingCancelled() {
            print("\u{001B}[2K\r❌ Model loading cancelled")
        }
        
        func loadingFailed(error: Error) {
            print("\u{001B}[2K\r❌ Model loading failed: \(error)")
        }
    }
    
    // Signal handling for cancellation
    private func setupSignalHandling(for modelLoader: ModelLoader) {
        // Create a signal source that will be triggered on SIGINT (Ctrl+C)
        let sigintSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        
        // Store the signal source in a property that will live for the duration of the command
        var signalSources: [DispatchSourceSignal] = []
        signalSources.append(sigintSource)
        
        // Set up the handler
        sigintSource.setEventHandler {
            print("\nCancelling model loading...")
            Task {
                await modelLoader.cancelLoading()
            }
        }
        
        // Register for the signal
        signal(SIGINT, SIG_IGN)
        sigintSource.resume()
        
        // Store the signal sources in a way they won't be deallocated
        objc_setAssociatedObject(
            self, 
            "signalSources", 
            signalSources, 
            .OBJC_ASSOCIATION_RETAIN
        )
    }
    
    mutating func run() async throws {
        // Load config
        let config = try YAMLConfig.load(from: meta)
        
        // Determine effective max tokens
        let effectiveMaxTokens: Int
        
        if let specifiedMaxTokens = maxTokens {
            effectiveMaxTokens = specifiedMaxTokens  // Use user-specified value
        } else if config.contextLength > 0 {
            effectiveMaxTokens = config.contextLength  // Use context length from config
        } else {
            effectiveMaxTokens = 512  // Default value if context length is unknown
        }
        
        // Initialize tokenizer with debug level and template
        print("\nInitializing tokenizer...")
        let tokenizer = try await Tokenizer(
            modelPath: config.tokenizerModel,
            template: template,
            debugLevel: debugLevel
        )
        
        // Create progress delegate
        let progressDelegate = CLIProgressDelegate(showDetailedProgress: showLoadingProgress)
        
        // Load models with progress reporting
        print("\nLoading models...")
        let modelLoader = ModelLoader(progressDelegate: progressDelegate)
        
        // Set up signal handling for cancellation
        setupSignalHandling(for: modelLoader)
        
        // Start model loading
        let modelLoadingTask = Task {
            try await modelLoader.loadModel(from: config)
        }
        
        // Wait for model loading to complete
        let models: LoadedModels
        do {
            models = try await modelLoadingTask.value
        } catch is CancellationError {
            print("Model loading was cancelled by user.")
            throw ExitCode.failure
        } catch {
            print("Failed to load models: \(error)")
            throw error
        }
        
        // Create inference manager with debug level
        let inferenceManager = try InferenceManager(
            models: models,
            contextLength: config.contextLength,
            batchSize: config.batchSize,
            splitLMHead: config.splitLMHead,
            debugLevel: debugLevel,
            v110: config.configVersion == "0.1.1"  // Set v110 flag based on version
        )
        
        if let prompt = prompt {
            // Initialize token printer
            let tokenPrinter = await TokenPrinter(
                tokenizer: tokenizer, 
                debugLevel: debugLevel,
                showSpecialTokens: showSpecialTokens
            )
            let generationStartTime = CFAbsoluteTimeGetCurrent()
            
            // Single prompt mode
            print("\nProcessing prompt: \"\(prompt)\"")
            
            var messages: [Tokenizer.ChatMessage] = []
            if let system = system {
                messages.append(Tokenizer.ChatMessage.assistant("I am an AI assistant. \(system)"))
            }
            messages.append(Tokenizer.ChatMessage.user(prompt))
            
            // Tokenize with template
            let tokens = tokenizer.applyChatTemplate(
                input: messages, 
                addGenerationPrompt: !noGenerationPrompt
            )
            if debugLevel >= 1 {
                print("Raw prompt:", prompt)
                print("Tokenized prompt:", tokenizer.decode(tokens: tokens))
                print("Token count:", tokens.count)
            }
            
            print("Assistant:", terminator: " ")
            
            // Run generation with token callback
            let (generatedTokens, prefillTime, stopReason) = try await inferenceManager.generateResponse(
                initialTokens: tokens,
                temperature: temperature,
                maxTokens: effectiveMaxTokens,
                eosTokens: tokenizer.eosTokenIds,
                tokenizer: tokenizer,
                onToken: { token in
                    Task {
                        await tokenPrinter.addToken(token)
                    }
                }
            )
            
            // Wait for printer to finish and get response
            let response = await tokenPrinter.stop()  // Get the response text
            await tokenPrinter.drain()  // Make sure everything is printed
            
            // Save response to file if requested
            if let saveFile = save {
                do {
                    try response.write(toFile: saveFile, atomically: true, encoding: .utf8)
                    print("\n\u{001B}[34mResponse saved to file: \(saveFile)\u{001B}[0m")
                } catch {
                    print("\n\u{001B}[31mError saving to file: \(error.localizedDescription)\u{001B}[0m")
                }
            }
            
            let inferenceEndTime = CFAbsoluteTimeGetCurrent()
            let inferenceTime = (inferenceEndTime - generationStartTime) - prefillTime
            let prefillMs = prefillTime * 1000
            
            let inferenceTokensPerSec = Double(generatedTokens.count) / inferenceTime
            let prefillTokensPerSec = Double(tokens.count) / prefillTime
            
            print("\n\u{001B}[34m\(String(format: "%.1f", inferenceTokensPerSec)) t/s, " +
                  "TTFT: \(String(format: "%.1f", prefillMs))ms " +
                  "(\(String(format: "%.1f", prefillTokensPerSec)) t/s), " +
                  "\(generatedTokens.count) tokens" +
                  " [Stop reason: \(stopReason)]\u{001B}[0m")
            
            // Add token ID output for debug level > 0 and prompt mode
            if debugLevel > 0 {
                print("\n=== TOKEN IDS ===")
                print("INPUT_TOKENS:", tokens.map(String.init).joined(separator: ","))
                print("OUTPUT_TOKENS:", generatedTokens.map(String.init).joined(separator: ","))
                print("STOP_REASON:", stopReason)
                
                // Try to include the stop token if it was EOS
                if stopReason == "eos" {
                    print("STOP_TOKENS:", tokenizer.eosTokenIds)
                    // Also try to get the text representation of the stop tokens
                    for eosId in tokenizer.eosTokenIds {
                        let stopTokenText = tokenizer.decode(tokens: [eosId], skipSpecialTokens: false)
                        print("STOP_TOKEN_TEXT for \(eosId):", stopTokenText)
                    }
                }
            }
        } else {
            // Interactive chat mode
            print("Context length: \(config.contextLength)")
            print("Max tokens: \(effectiveMaxTokens)")
            print("Prefill batch size: \(config.batchSize)")
            print("\nStarting interactive chat. Press Ctrl+D to exit.")
            print("Type your message and press Enter to chat. Use /t to toggle thinking mode.")
            print("Thinking mode is \(thinkingMode ? "ON" : "OFF")")
            
            var conversation: [Tokenizer.ChatMessage] = []
            
            if let system = system {
                conversation.append(.assistant("I am an AI assistant. \(system)"))
            }
            
            // Initialize token printer outside the loop
            let tokenPrinter = await TokenPrinter(
                tokenizer: tokenizer, 
                debugLevel: debugLevel,
                showSpecialTokens: showSpecialTokens
            )
            
            while true {
                await tokenPrinter.drain()
                
                print("\n\u{001B}[92mYou:\u{001B}[0m", terminator: " ")
                fflush(stdout)
                
                // Read and clean input
                guard let rawInput = readLine(strippingNewline: true) else { 
                    break 
                }
                
                // Clean the input
                let input = rawInput
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .replacingOccurrences(of: "assistant", with: "")
                    .replacingOccurrences(of: "*", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                
                if input.isEmpty { continue }
                
                // Handle /t command
                if input == "/t" {
                    thinkingMode.toggle()
                    print("Thinking mode \(thinkingMode ? "ON" : "OFF")")
                    continue  // Skip adding to conversation
                }
                
                // Before adding new message, check if we need to trim history
                let maxContextSize = config.contextLength - 200  // Leave more room for response
                
                // Add new message and check context size
                if !input.starts(with: "/") {  // Only add non-command inputs to conversation
                    conversation.append(.user(input))
                    let currentTokens = tokenizer.applyChatTemplate(
                        input: conversation,
                        addGenerationPrompt: !noGenerationPrompt
                    )
                    
                    // Trim history until we fit in context
                    while currentTokens.count > maxContextSize && conversation.count > 1 {  // Changed from > 2
                        if debugLevel >= 1 {
                            print("Trimming conversation history to fit context...")
                        }
                        conversation.removeFirst(1)  // Remove one message at a time
                        let newTokens = tokenizer.applyChatTemplate(
                            input: conversation,
                            addGenerationPrompt: !noGenerationPrompt
                        )
                        if newTokens.count <= maxContextSize {
                            if debugLevel >= 1 {
                                print("New context size:", newTokens.count)
                            }
                            break
                        }
                    }
                }
                
                // Apply final template with thinking mode if enabled
                var messages = conversation
                if thinkingMode {
                    messages.insert(.assistant(Self.THINKING_PROMPT), at: 0)
                }
                
                let tokens = tokenizer.applyChatTemplate(
                    input: messages,
                    addGenerationPrompt: !noGenerationPrompt
                )
                
                if debugLevel >= 1 {
                    print("\nContext before generation:")
                    print("Total tokens:", tokens.count)
                    print("Context length limit:", config.contextLength)
                    print("Messages count:", messages.count)
                }
                
                // Before generating response
                print("\nAssistant:", terminator: " ")
                await tokenPrinter.startNewMessage()  // Reset state for new message
                let generationStartTime = CFAbsoluteTimeGetCurrent()
                
                let (generatedTokens, prefillTime, stopReason) = try await inferenceManager.generateResponse(
                    initialTokens: tokens,
                    temperature: temperature,
                    maxTokens: effectiveMaxTokens,
                    eosTokens: tokenizer.eosTokenIds,
                    tokenizer: tokenizer,
                    onToken: { token in
                        Task {
                            await tokenPrinter.addToken(token)
                        }
                    }
                )
                
                // Wait for printer to finish and get response
                let response = await tokenPrinter.stop()  // Keep the response for conversation
                conversation.append(.assistant(response))
                await tokenPrinter.drain()
                
                // Save response to file if requested (in chat mode)
                if let saveFile = save {
                    do {
                        try response.write(toFile: saveFile, atomically: true, encoding: .utf8)
                        print("\n\u{001B}[34mResponse saved to file: \(saveFile)\u{001B}[0m")
                    } catch {
                        print("\n\u{001B}[31mError saving to file: \(error.localizedDescription)\u{001B}[0m")
                    }
                }
                
                // Now print stats with stop reason
                let inferenceEndTime = CFAbsoluteTimeGetCurrent()
                let inferenceTime = (inferenceEndTime - generationStartTime) - prefillTime
                let prefillMs = prefillTime * 1000
                
                let inferenceTokensPerSec = Double(generatedTokens.count) / inferenceTime
                let prefillTokensPerSec = Double(tokens.count) / prefillTime
                
                print("\n\u{001B}[34m\(String(format: "%.1f", inferenceTokensPerSec)) t/s, " +
                      "TTFT: \(String(format: "%.1f", prefillMs))ms " +
                      "(\(String(format: "%.1f", prefillTokensPerSec)) t/s), " +
                      "\(generatedTokens.count) tokens" +
                      " [Stop reason: \(stopReason)]\u{001B}[0m")
            }
        }
    }
}