import XCTest
@testable import AnemllCore

final class TokenizerTests: XCTestCase {
    func testTokenizer() throws {
        // Replace with your model's metadata path
        let modelPath = "/path/to/your/model/metadata"
        
        // Initialize tokenizer
        let tokenizer = try Tokenizer(modelPath: modelPath)
        
        // Test encoding
        let text = "Hello, world!"
        let tokens = tokenizer.tokenize(text)
        print("Encoded tokens:", tokens)
        
        // Test decoding
        let decoded = tokenizer.detokenize(tokens)
        print("Decoded text:", decoded)
        
        XCTAssertEqual(decoded, text)
    }
} 