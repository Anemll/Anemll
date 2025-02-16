import CoreML

/// Represents a single FFN chunk that provides both prefill and infer functions.
public struct FFNChunk {
    public let inferModel: MLModel
    public let prefillModel: MLModel
} 