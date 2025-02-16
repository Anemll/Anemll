import Foundation
import Tokenizers
import Hub
import CoreML

/// Wraps a tokenizer from the swift-transformers package.
public final class Tokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizers.Tokenizer
    public let eosTokenId: Int  // Changed from var to let
    private let debug: Bool = true  // Add debug property
    private let debugLevel: Int
    private let template: String  // Add template property
    
    /// Initializes the tokenizer with a local model path.
    ///
    /// - Parameter modelPath: The path to the tokenizer files
    /// - Throws: TokenizerError if initialization fails
    public init(modelPath: String, template: String = "default", debugLevel: Int = 0) async throws {
        self.debugLevel = debugLevel
        self.template = template
        print("\nTokenizer Debug:")
        print("Input modelPath: \(modelPath)")
        print("Using template: \(template)")
        
        // Create URL from path
        let modelURL = URL(fileURLWithPath: modelPath)
        print("Using modelURL: \(modelURL.path)")
        
        // List all files in directory
        let fileManager = FileManager.default
        if let files = try? fileManager.contentsOfDirectory(atPath: modelPath) {
            print("\nFiles in directory:")
            for file in files {
                print("- \(file)")
            }
        }
        
        // Create minimal config.json if it doesn't exist
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
        
        // Load tokenizer
        print("\nAttempting to load tokenizer...")
        do {
            self.tokenizer = try await AutoTokenizer.from(
                modelFolder: modelURL
            )
            
            // Get EOS token ID based on template
            let eosToken = template == "deephermes" ? "<|endoftext|>" : "</s>"
            let eosTokens = tokenizer.encode(text: eosToken)
            if let eos = eosTokens.first {
                self.eosTokenId = eos
                print("✓ EOS token ID: \(eos) (\(eosToken))")
            } else {
                throw TokenizerError.initializationFailed("Could not find EOS token ID")
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

    public struct ChatMessage {
        public let role: String
        public let content: String
        
        public static func user(_ content: String) -> ChatMessage {
            ChatMessage(role: "user", content: content)
        }
        
        public static func assistant(_ content: String) -> ChatMessage {
            ChatMessage(role: "assistant", content: content)
        }
    }

    // Consolidated applyChatTemplate method
    public func applyChatTemplate(input: Any, addGenerationPrompt: Bool = true) -> [Int] {
        var formatted: String = ""

        // Skip template when addGenerationPrompt is false
        if !addGenerationPrompt {
            if let text = input as? String {
                formatted = "<s>" + text
            } 
            return tokenize(formatted)
        }

        // Apply template when addGenerationPrompt is true
        if let text = input as? String {
            formatted += "Human: \(text)\n"
            formatted += "Assistant:"
        } else if let messages = input as? [ChatMessage] {
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
                // Add fallback formatting if chat template fails
                for message in messages {
                    if message.role == "user" {
                        formatted += "Human: \(message.content)\n"
                    } else if message.role == "assistant" {
                        formatted += "Assistant: \(message.content)\n"
                    }
                }
                formatted += "Assistant:"
            }       
        }
        
        formatted = "<s>" + formatted
        
        if debugLevel >= 2 {
            print("\nApplying chat template:")
            print("Formatted: \"\(formatted)\"")
        }
        
        return tokenize(formatted)
    }

    // Method to decode tokens back to text
    public func decode(tokens: [Int], skipSpecialTokens: Bool = true) -> String {
        return tokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
    }
}

/// Errors that can occur during tokenization
public enum TokenizerError: Error {
    case initializationFailed(String)
    case tokenizationFailed(String)
    case decodingFailed(String)
} 