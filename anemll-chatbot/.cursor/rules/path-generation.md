# Path Generation Rule

@files: **/Services/InferenceService.swift, **/Services/ModelService.swift, **/AnemllCore/**

## Rule: Use AnemllCore for Path Generation

### Description
Always use the path generation logic from AnemllCore (specifically YAMLConfig) instead of implementing similar logic in the chatbot application.

### Applies To
- Any code that constructs model file paths
- Files that involve model loading or verification
- Services that interact with ML models

### Rationale
Duplicating path generation logic leads to inconsistencies, harder maintenance, and potential bugs. The canonical implementation in AnemllCore should be the single source of truth.

### Correct Implementation
```swift
// Pass metadata to YAMLConfig
let yamlDict: [String: Any] = [
    "model_path": basePath,
    "tokenizer_model": tokenizerPath,
    "context_length": contextLength,
    "batch_size": batchSize,
    "model_prefix": modelPrefix,
    "lut_ffn": lutFFN,
    "lut_lmhead": lutLMHead,
    "lut_embeddings": lutEmbeddings,
    "num_chunks": numChunks
]

// Let YAMLConfig generate canonical paths
let yamlString = try Yams.dump(object: yamlDict)
let config = try YAMLConfig(from: yamlString)

// Use the paths provided by YAMLConfig
let embedPath = config.embedPath
let lmheadPath = config.lmheadPath
let ffnPath = config.ffnPath
```

### Incorrect Implementation
```swift
// ❌ Don't do this - duplicate path generation logic
func generateModelPaths(modelPrefix: String, lutFFN: Int, numChunks: Int) -> String {
    // This duplicates logic that exists in YAMLConfig
    return "/path/to/\(modelPrefix)_FFN_PF_lut\(lutFFN)_chunk_01of\(String(format: "%02d", numChunks)).mlmodelc"
}
```

### Related Context
@file: ../../ARCHITECTURE.md

### Enforcement
- During code reviews, check for any custom path generation logic that duplicates YAMLConfig functionality
- When modifying existing non-compliant code, refactor to use YAMLConfig
- Report violations of this rule to the project maintainers 