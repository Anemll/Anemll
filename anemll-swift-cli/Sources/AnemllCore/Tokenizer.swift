import Foundation
import Tokenizers
import Hub
import CoreML

/// Wraps a tokenizer from the swift-transformers package.
public final class Tokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizers.Tokenizer
    public let eosTokenId: Int
    public let bosTokenId: Int  // Add BOS token ID property
    public let padTokenId: Int  // Add PAD token ID property
    private let debug: Bool = true
    private let debugLevel: Int
    private var chatTemplate: String? // Add chat template property

    public init(modelPath: String, template: String = "default", debugLevel: Int = 0) async throws {
        self.debugLevel = debugLevel
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

            // Define a variable to hold the EOS token
            var eosToken = "</s>"  // Default value
            // Define a variable to hold the BOS token
            var bosToken = "<s>"   // Default value
            // Define a variable to hold the PAD token
            var padToken = "<pad>"  // Default value

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
                        "chatglm": "</s>"
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
                        "chatglm": "<s>"
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
                        "chatglm": "<pad>"
                    ]
                    
                    if let templateToken = padTokenMap[template] {
                        padToken = templateToken
                        print("Using template-specific PAD token: \(padToken) for template: \(template)")
                    } else {
                        print("Using default PAD token: \(padToken)")
                    }
                }
            }

            // Now encode the correct EOS token
            let eosTokens = tokenizer.encode(text: eosToken)
            if let eos = eosTokens.first {
                self.eosTokenId = eos
                print("✓ EOS token ID: \(eos) for token '\(eosToken)'")
            } else {
                throw TokenizerError.initializationFailed("Could not find EOS token ID for '\(eosToken)'")
            }

            // Now encode the correct BOS token
            let bosTokens = tokenizer.encode(text: bosToken)
            if let bos = bosTokens.first {
                self.bosTokenId = bos
                print("✓ BOS token ID: \(bos) for token '\(bosToken)'")
            } else {
                throw TokenizerError.initializationFailed("Could not find BOS token ID for '\(bosToken)'")
            }

            // Now encode the correct PAD token
            let padTokens = tokenizer.encode(text: padToken)
            if let pad = padTokens.first {
                self.padTokenId = pad
                print("✓ PAD token ID: \(pad) for token '\(padToken)'")
            } else {
                throw TokenizerError.initializationFailed("Could not find PAD token ID for '\(padToken)'")
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
        // Skip template when addGenerationPrompt is false
        if !addGenerationPrompt {
            if let text = input as? String {
                return tokenize("" + text)
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
                print("Error applying chat template: \(error)")
                // Fallback: use simple prompt formatting for complex templates
                print("Using fallback prompt formatting...")
                if let userMessage = messagesArray.first(where: { $0["role"] as? String == "user" }),
                   let content = userMessage["content"] as? String {
                    let formattedPrompt = "<|im_start|>user\n\(content)<|im_end|>\n<|im_start|>assistant\n"
                    return tokenizer.encode(text: formattedPrompt)
                }
                return []
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
