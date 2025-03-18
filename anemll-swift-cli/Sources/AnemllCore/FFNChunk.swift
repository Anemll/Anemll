@preconcurrency import CoreML

/// Represents a single FFN chunk that provides both prefill and infer functions.
public struct FFNChunk: @unchecked Sendable {
    public let inferModel: MLModel
    public let prefillModel: MLModel
    
    public init(inferModel: MLModel, prefillModel: MLModel) {
        self.inferModel = inferModel
        self.prefillModel = prefillModel
    }
} 