# Error Handling Rule

@files: **/Services/*.swift, **/Models/*.swift

## Rule: Structured Error Handling

### Description
Use consistent, structured error handling throughout the codebase with descriptive messages and proper propagation.

### Applies To
- Any code that can throw errors
- Services that handle external operations
- Model validation and loading

### Rationale
Consistent error handling makes debugging easier, improves user experience with clear error messages, and creates a more maintainable codebase.

### Correct Implementation
```swift
// Define custom error types
enum InferenceError: Error, LocalizedError {
    case modelPathNotFound
    case invalidConfig
    case inferenceError(String)
    case tokenizerError(String)
    case modelLoadingCancelled
    
    var errorDescription: String? {
        switch self {
        case .modelPathNotFound:
            return "Model directory was not found."
        case .invalidConfig:
            return "Invalid model configuration."
        case .inferenceError(let message):
            return "Inference error: \(message)"
        case .tokenizerError(let message):
            return "Tokenizer error: \(message)"
        case .modelLoadingCancelled:
            return "Model loading was cancelled."
        }
    }
}

// Proper error throwing with context
func loadModel() throws {
    guard FileManager.default.fileExists(atPath: modelPath) else {
        print("Error: Model path not found at \(modelPath)")
        throw InferenceError.modelPathNotFound
    }
    // ...
}

// Graceful error handling
do {
    try loadModel()
} catch InferenceError.modelPathNotFound {
    // Specific handling for this error
    showModelNotFoundAlert()
} catch {
    // Generic error handling with logging
    print("Failed to load model: \(error.localizedDescription)")
    showGenericErrorAlert(message: error.localizedDescription)
}
```

### Incorrect Implementation
```swift
// ❌ Don't use generic errors without context
func loadModel() throws {
    if !FileManager.default.fileExists(atPath: modelPath) {
        throw NSError(domain: "ModelError", code: 1, userInfo: nil)
    }
    // ...
}

// ❌ Don't handle errors without providing feedback
do {
    try loadModel()
} catch {
    // Missing logging and user feedback
    print("Error occurred")
}
```

### Related Context
@file: ../../ARCHITECTURE.md

### Enforcement
- All functions that can fail should throw appropriate custom errors
- Error messages must include sufficient context (e.g., file paths, relevant values)
- Log errors before propagating them up the stack
- Ensure errors are eventually communicated to the user in a friendly way 