import Foundation
import Tokenizers
import Hub
import CoreML

/// Wraps a tokenizer from the swift-transformers package.
public final class Tokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizers.Tokenizer
    public let eosTokenIds: [Int]  // Changed to array to support multiple EOS tokens
    public let bosTokenId: Int  // Add BOS token ID property
    public let padTokenId: Int  // Add PAD token ID property
    private let debug: Bool = true
    private let debugLevel: Int
    private var chatTemplate: String? // Add chat template property
    private let templateName: String  // Store template name for fallback formatting

    public init(modelPath: String, template: String = "default", debugLevel: Int = 0) async throws {
        self.debugLevel = debugLevel
        self.templateName = template
        print("\nTokenizer Debug:")
        print("Input modelPath: \(modelPath)")
        print("Using template: \(template)")

        let modelURL = URL(fileURLWithPath: modelPath)
        print("Using modelURL: \(modelURL.path)")

        let fileManager = FileManager.default
        if let files = try? fileManager.contentsOfDirectory(atPath: modelPath) {
            print("\nFiles in directory:")
            for file in files {
                print("- \(file)")
            }
        }

        let configPath = modelURL.appendingPathComponent("config.json")
        if !fileManager.fileExists(atPath: configPath.path) {
            print("\nCreating minimal config.json...")
            let configDict: [String: Any] = [
                "model_type": "llama",
                "tokenizer_class": "LlamaTokenizer"
            ]
            let configData = try JSONSerialization.data(withJSONObject: configDict, options: .prettyPrinted)
            try configData.write(to: configPath)
            print("Created config.json at: \(configPath.path)")
        }

        print("\nChecking specific files:")
        print("config.json exists: \(fileManager.fileExists(atPath: configPath.path))")
        print("tokenizer_config.json exists: \(fileManager.fileExists(atPath: modelURL.appendingPathComponent("tokenizer_config.json").path))")
        print("tokenizer.json exists: \(fileManager.fileExists(atPath: modelURL.appendingPathComponent("tokenizer.json").path))")

        print("\nAttempting to load tokenizer...")
        do {
            self.tokenizer = try await AutoTokenizer.from(
                modelFolder: modelURL
            )

            // Load tokenizer_config.json
            let tokenizerConfigPath = modelURL.appendingPathComponent("tokenizer_config.json")
            var tokenizerConfig: [String: Any]? = nil  // Declare at this scope level
            
            if fileManager.fileExists(atPath: tokenizerConfigPath.path) {
                print("Loading tokenizer_config.json")
                let tokenizerConfigData = try Data(contentsOf: tokenizerConfigPath)
                tokenizerConfig = try JSONSerialization.jsonObject(with: tokenizerConfigData) as? [String: Any]
                
                if let config = tokenizerConfig {
                    if let chatTemplate = config["chat_template"] as? String {
                        self.chatTemplate = chatTemplate
                        print("Found chat_template in tokenizer_config.json: \(chatTemplate)")
                    } else {
                        print("No chat_template found in tokenizer_config.json, using default")
                    }
                }
            } else {
                print("tokenizer_config.json not found.")
            }

            // Define variables to hold the tokens
            var eosToken = "</s>"  // Default value
            var bosToken = "<s>"   // Default value
            var padToken = "<pad>"  // Default value
            var eosTokenIdsList: [Int] = []  // To store multiple EOS token IDs
            var bosTokenIdFromJson: Int? = nil
            var padTokenIdFromJson: Int? = nil

            // First, try to get eos_token_id from config.json
            var configJson: [String: Any]? = nil
            if fileManager.fileExists(atPath: configPath.path) {
                let configData = try Data(contentsOf: configPath)
                configJson = try JSONSerialization.jsonObject(with: configData) as? [String: Any]

                // Check for eos_token_id (can be single Int or array of Ints)
                if let eosId = configJson?["eos_token_id"] {
                    if let singleId = eosId as? Int {
                        eosTokenIdsList = [singleId]
                        print("Found single eos_token_id in config.json: \(singleId)")
                    } else if let multipleIds = eosId as? [Int] {
                        eosTokenIdsList = multipleIds
                        print("Found multiple eos_token_ids in config.json: \(multipleIds)")
                    }
                }

                // Check for bos_token_id
                if let bosId = configJson?["bos_token_id"] as? Int {
                    bosTokenIdFromJson = bosId
                    print("Found bos_token_id in config.json: \(bosId)")
                }

                // Check for pad_token_id
                if let padId = configJson?["pad_token_id"] as? Int {
                    padTokenIdFromJson = padId
                    print("Found pad_token_id in config.json: \(padId)")
                }
            }

            // Try to read token IDs from tokenizer.json added_tokens array
            // This is more reliable than encoding token strings for special tokens
            let tokenizerJsonPath = modelURL.appendingPathComponent("tokenizer.json")
            if fileManager.fileExists(atPath: tokenizerJsonPath.path) {
                do {
                    let tokenizerJsonData = try Data(contentsOf: tokenizerJsonPath)
                    if let tokenizerJson = try JSONSerialization.jsonObject(with: tokenizerJsonData) as? [String: Any],
                       let addedTokens = tokenizerJson["added_tokens"] as? [[String: Any]] {
                        print("Reading special token IDs from tokenizer.json added_tokens...")
                        for tokenEntry in addedTokens {
                            if let id = tokenEntry["id"] as? Int,
                               let content = tokenEntry["content"] as? String {
                                // Match common EOS token patterns - collect ALL matching tokens
                                if content == "<eos>" || content == "</s>" || content == "<|endoftext|>" || content == "<end_of_turn>" || content == "<|im_end|>" || content == "<|eot_id|>" {
                                    if !eosTokenIdsList.contains(id) {
                                        eosTokenIdsList.append(id)
                                        print("  Found EOS token in added_tokens: '\(content)' = \(id)")
                                    }
                                }
                                // Match common BOS token patterns
                                if content == "<bos>" || content == "<s>" || content == "<|startoftext|>" {
                                    if bosTokenIdFromJson == nil {
                                        bosTokenIdFromJson = id
                                        bosToken = content
                                        print("  Found BOS token in added_tokens: '\(content)' = \(id)")
                                    }
                                }
                                // Match common PAD token patterns
                                if content == "<pad>" || content == "<|padding|>" {
                                    if padTokenIdFromJson == nil {
                                        padTokenIdFromJson = id
                                        padToken = content
                                        print("  Found PAD token in added_tokens: '\(content)' = \(id)")
                                    }
                                }
                            }
                        }
                    }
                } catch {
                    print("Warning: Could not parse tokenizer.json: \(error)")
                }
            }

            // Try to get EOS token from tokenizer_config.json
            if let config = tokenizerConfig {
                // First try to access eos_token as a dictionary (which it seems to be from the screenshot)
                if let eosTokenObj = config["eos_token"] as? [String: Any],
                   let content = eosTokenObj["content"] as? String {
                    eosToken = content  // Get the content value from the eos_token object
                    print("Found EOS token object in tokenizer_config.json with content: \(eosToken)")
                } 
                // Fallback to direct string access (the original implementation)
                else if let eos_token = config["eos_token"] as? String {
                    eosToken = eos_token
                    print("Found EOS token in tokenizer_config.json: \(eosToken)")
                } else {
                    print("Not found EOS token in tokenizer_config.json: \(eosToken)")

                    // Fallback to template-based mapping
                    let eosTokenMap: [String: String] = [
                        "default": "</s>",
                        "deepseek": "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>",  // do not change, this is correct fo DS R1
                        "deephermes": "<|im_end|>",
                        "llama": "</s>",
                        "mistral": "</s>",
                        "falcon": "</s>",
                        "chatglm": "</s>",
                        "gemma": "<end_of_turn>",
                        "gemma3": "<end_of_turn>"
                    ]
                    
                    if let templateToken = eosTokenMap[template] {
                        eosToken = templateToken
                        print("Using template-specific EOS token: \(eosToken) for template: \(template)")
                    } else {
                        print("Using default EOS token: \(eosToken)")
                    }
                }
                
                // Try to get BOS token from tokenizer_config.json
                if let bosTokenObj = config["bos_token"] as? [String: Any],
                   let content = bosTokenObj["content"] as? String {
                    bosToken = content  // Get the content value from the bos_token object
                    print("Found BOS token object in tokenizer_config.json with content: \(bosToken)")
                } 
                // Fallback to direct string access
                else if let bos_token = config["bos_token"] as? String {
                    bosToken = bos_token
                    print("Found BOS token in tokenizer_config.json: \(bosToken)")
                } else {
                    print("Not found BOS token in tokenizer_config.json: \(bosToken)")

                    // Fallback to template-based mapping
                    let bosTokenMap: [String: String] = [
                        "default": "<s>",
                        "deepseek": "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>",
                        "deephermes": "<|im_start|>",
                        "llama": "<s>",
                        "mistral": "<s>",
                        "falcon": "<s>",
                        "chatglm": "<s>",
                        "gemma": "<bos>",
                        "gemma3": "<bos>"
                    ]
                    
                    if let templateToken = bosTokenMap[template] {
                        bosToken = templateToken
                        print("Using template-specific BOS token: \(bosToken) for template: \(template)")
                    } else {
                        print("Using default BOS token: \(bosToken)")
                    }
                }

                // Try to get PAD token from tokenizer_config.json
                if let padTokenObj = config["pad_token"] as? [String: Any],
                   let content = padTokenObj["content"] as? String {
                    padToken = content  // Get the content value from the pad_token object
                    print("Found PAD token object in tokenizer_config.json with content: \(padToken)")
                } 
                // Fallback to direct string access
                else if let pad_token = config["pad_token"] as? String {
                    padToken = pad_token
                    print("Found PAD token in tokenizer_config.json: \(padToken)")
                } else {
                    print("Not found PAD token in tokenizer_config.json: \(padToken)")

                    // Fallback to template-based mapping
                    let padTokenMap: [String: String] = [
                        "default": "<pad>",
                        "deepseek": "<pad>",
                        "deephermes": "<|padding|>",
                        "llama": "<pad>",
                        "mistral": "<pad>",
                        "falcon": "<pad>",
                        "chatglm": "<pad>",
                        "gemma": "<pad>",
                        "gemma3": "<pad>"
                    ]
                    
                    if let templateToken = padTokenMap[template] {
                        padToken = templateToken
                        print("Using template-specific PAD token: \(padToken) for template: \(template)")
                    } else {
                        print("Using default PAD token: \(padToken)")
                    }
                }
            }

            // If we didn't get EOS token IDs from config.json or tokenizer.json, encode the EOS token string
            if eosTokenIdsList.isEmpty {
                let eosTokens = tokenizer.encode(text: eosToken)
                if let eos = eosTokens.first {
                    eosTokenIdsList = [eos]
                    print("✓ EOS token ID (from encode): \(eos) for token '\(eosToken)'")
                } else {
                    throw TokenizerError.initializationFailed("Could not find EOS token ID for '\(eosToken)'")
                }
            }

            // Explicitly look up critical stop tokens (like Python's build_stop_token_ids)
            // This ensures we catch <end_of_turn> even if it's not in added_tokens
            let extraStopTokens = ["<end_of_turn>", "<|eot_id|>", "<|endoftext|>", "<|im_end|>"]
            for stopToken in extraStopTokens {
                let encoded = tokenizer.encode(text: stopToken)
                // Only add if encoding produces a single token (not a sequence)
                if encoded.count == 1, let tokenId = encoded.first, !eosTokenIdsList.contains(tokenId) {
                    eosTokenIdsList.append(tokenId)
                    print("✓ Added stop token (from encode): '\(stopToken)' = \(tokenId)")
                }
            }

            // Set the EOS token IDs
            self.eosTokenIds = eosTokenIdsList
            print("✓ Using EOS token IDs: \(eosTokenIdsList)")

            // Use BOS token ID from JSON if available, otherwise encode
            if let bos = bosTokenIdFromJson {
                self.bosTokenId = bos
                print("✓ BOS token ID (from JSON): \(bos) for token '\(bosToken)'")
            } else {
                let bosTokens = tokenizer.encode(text: bosToken)
                if let bos = bosTokens.first {
                    self.bosTokenId = bos
                    print("✓ BOS token ID (from encode): \(bos) for token '\(bosToken)'")
                } else {
                    throw TokenizerError.initializationFailed("Could not find BOS token ID for '\(bosToken)'")
                }
            }

            // Use PAD token ID from JSON if available, otherwise encode
            if let pad = padTokenIdFromJson {
                self.padTokenId = pad
                print("✓ PAD token ID (from JSON): \(pad) for token '\(padToken)'")
            } else {
                let padTokens = tokenizer.encode(text: padToken)
                if let pad = padTokens.first {
                    self.padTokenId = pad
                    print("✓ PAD token ID (from encode): \(pad) for token '\(padToken)'")
                } else {
                    throw TokenizerError.initializationFailed("Could not find PAD token ID for '\(padToken)'")
                }
            }

            print("✓ Tokenizer loaded successfully!")
        } catch {
            print("✗ Failed to load tokenizer: \(error)")
            throw TokenizerError.initializationFailed("Failed to load tokenizer: \(error)")
        }
    }
    
    /// Tokenizes the given text into token IDs.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: Array of token IDs
    /// - Throws: TokenizerError if tokenization fails
    public func tokenize(_ text: String) -> [Int] {
        return tokenizer.encode(text: text)
    }
    
    /// Converts token IDs back to a string.
    ///
    /// - Parameter tokens: Array of token IDs to decode
    /// - Returns: The decoded text
    /// - Throws: TokenizerError if decoding fails
    public func detokenize(_ tokens: [Int]) -> String {
        return tokenizer.decode(tokens: tokens)
    }

    public struct ChatMessage: Sendable {
        public let role: String
        public let content: String
        
        public static func user(_ content: String) -> ChatMessage {
            ChatMessage(role: "user", content: content)
        }
        
        public static func assistant(_ content: String) -> ChatMessage {
            ChatMessage(role: "assistant", content: content)
        }
        public static func system(_ content: String) -> ChatMessage {
            ChatMessage(role: "system", content: content)
        }
    }

    // Consolidated applyChatTemplate method
    public func applyChatTemplate(input: Any, addGenerationPrompt: Bool = true) -> [Int] {
        // When addGenerationPrompt is false, tokenize the raw content without template
        if !addGenerationPrompt {
            if let text = input as? String {
                return tokenize(text)
            } else if let messages = input as? [ChatMessage] {
                // Concatenate all message contents and tokenize
                let combinedText = messages.map { $0.content }.joined(separator: " ")
                return tokenize(combinedText)
            }
            return []
        }

        if let messages = input as? [ChatMessage] {
            // Convert ChatMessage instances to the expected format
            let messagesArray = messages.map { message in
                return ["role": message.role, "content": message.content]
            }
            do {
                let tokens = try tokenizer.applyChatTemplate(messages: messagesArray)
                if debugLevel >= 1 {
                    print("\nTokens:", tokens)
                    print("Decoded:", tokenizer.decode(tokens: tokens))
                }
                return tokens
            } catch {
                if debugLevel >= 1 {
                    print("Error applying chat template: \(error)")
                    // Fallback: use template-specific prompt formatting for ALL messages (multi-turn)
                    print("Using fallback prompt formatting for template: \(templateName)")
                }

                let formattedPrompt: String
                switch templateName.lowercased() {
                case "gemma", "gemma3":
                    // Gemma format: <bos><start_of_turn>role\n{content}<end_of_turn>\n...
                    // Note: Gemma3 doesn't support "system" role - it uses "model" for system messages
                    var prompt = "<bos>"
                    for message in messagesArray {
                        let role = message["role"] as? String ?? "user"
                        let content = message["content"] as? String ?? ""
                        // Map both "assistant" and "system" to "model" for Gemma3
                        let gemmaRole = (role == "assistant" || role == "system") ? "model" : role
                        prompt += "<start_of_turn>\(gemmaRole)\n\(content)<end_of_turn>\n"
                    }
                    prompt += "<start_of_turn>model\n"
                    formattedPrompt = prompt

                case "llama", "llama3":
                    // LLaMA 3 format
                    var prompt = "<|begin_of_text|>"
                    for message in messagesArray {
                        let role = message["role"] as? String ?? "user"
                        let content = message["content"] as? String ?? ""
                        prompt += "<|start_header_id|>\(role)<|end_header_id|>\n\n\(content)<|eot_id|>"
                    }
                    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    formattedPrompt = prompt

                case "deepseek":
                    // DeepSeek format
                    var prompt = "<｜begin▁of▁sentence｜>"
                    for message in messagesArray {
                        let role = message["role"] as? String ?? "user"
                        let content = message["content"] as? String ?? ""
                        if role == "user" {
                            prompt += "User: \(content)\n\n"
                        } else if role == "assistant" {
                            prompt += "Assistant: \(content)\n\n"
                        } else if role == "system" {
                            prompt += "\(content)\n\n"
                        }
                    }
                    prompt += "Assistant:"
                    formattedPrompt = prompt

                case "qwen", "qwen2", "qwen3":
                    // Qwen/ChatML format
                    var prompt = ""
                    for message in messagesArray {
                        let role = message["role"] as? String ?? "user"
                        let content = message["content"] as? String ?? ""
                        prompt += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
                    }
                    prompt += "<|im_start|>assistant\n"
                    formattedPrompt = prompt

                default:
                    // Default ChatML format
                    var prompt = ""
                    for message in messagesArray {
                        let role = message["role"] as? String ?? "user"
                        let content = message["content"] as? String ?? ""
                        prompt += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
                    }
                    prompt += "<|im_start|>assistant\n"
                    formattedPrompt = prompt
                }

                if debugLevel >= 1 {
                    print("Fallback formatted prompt (\(messagesArray.count) messages): \(formattedPrompt.prefix(200))...")
                }
                return tokenizer.encode(text: formattedPrompt)
            }
        }

        // Default to empty array in case of non-array input
        return []
    }
    // Method to decode tokens back to text
    public func decode(tokens: [Int], skipSpecialTokens: Bool = true) -> String {
        return tokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
    }
}

// Extension to provide convenient role checking
extension Tokenizer.ChatMessage {
    public var isAssistant: Bool {
        return role == "assistant"
    }
    
    public var isUser: Bool {
        return role == "user"
    }
}

// Extension to make debugging easier
extension Tokenizer.ChatMessage: CustomStringConvertible {
    public var description: String {
        return "\(role)(\"\(content)\")"
    }
}

/// Errors that can occur during tokenization
public enum TokenizerError: Error {
    case initializationFailed(String)
    case tokenizationFailed(String)
    case decodingFailed(String)
} 
