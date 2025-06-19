// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// TokenPrinter.swift

import Foundation
import AnemllCore

/// Actor to ensure thread-safe access to the token buffer
@globalActor actor TokenPrinterActor {
    static let shared = TokenPrinterActor()
}

/// TokenPrinter handles the streaming of tokens from the model
/// It decodes tokens and maintains a buffer of the generated text
@TokenPrinterActor
class TokenPrinter {
    private let tokenizer: Tokenizer
    private var buffer: String = ""
    
    /// Initialize with a tokenizer
    /// - Parameter tokenizer: The tokenizer to use for decoding tokens
    init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }
    
    /// Validates that the tokenizer is properly initialized
    /// - Throws: An error if the tokenizer is not valid
    func validateTokenizer() throws {
        // This method is used to make a potential throwing point in the call chain
        if tokenizer.eosTokenIds.isEmpty || (tokenizer.eosTokenIds.first ?? -1) < 0 {
            throw InferenceError.inferenceError("Tokenizer has invalid EOS token ID")
        }
    }
    
    /// Add a token to the buffer
    /// - Parameter token: The token ID to decode and add
    func addToken(_ token: Int) async {
        let decoded = tokenizer.decode(tokens: [token])
        buffer += decoded
        
        // Print every 10th token to avoid flooding the console
        if buffer.count % 10 == 0 {
            //print("DEBUG - TokenPrinter buffer length: \(buffer.count) characters")
        }
    }
    
    /// Get the current buffer contents
    /// - Returns: The current text in the buffer
    func getBuffer() -> String {
        return buffer
    }
    
    /// Reset the buffer to empty
    func reset() {
        print("DEBUG - TokenPrinter buffer reset")
        buffer = ""
    }
    
    /// Stop generation and return the final buffer contents
    /// - Returns: The final generated text
    func stop() -> String {
        let response = buffer
        print("DEBUG - TokenPrinter stopped, final buffer length: \(buffer.count) characters")
        buffer = ""  // Reset for next use
        return response
    }
} 
