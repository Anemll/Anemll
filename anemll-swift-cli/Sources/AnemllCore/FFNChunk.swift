@preconcurrency import CoreML

/// Represents a single FFN chunk that provides both prefill and infer functions.
/// For Gemma3 models with sliding window attention (context > 512), also provides
/// rotation functions (infer_rotate, prefill_rotate) for cache rotation mode.
public struct FFNChunk: @unchecked Sendable {
    public let inferModel: MLModel
    public let prefillModel: MLModel

    // Optional rotation models for Gemma3 with sliding window (4-function models)
    public let inferRotateModel: MLModel?
    public let prefillRotateModel: MLModel?

    /// Standard 2-function initializer (infer, prefill)
    public init(inferModel: MLModel, prefillModel: MLModel) {
        self.inferModel = inferModel
        self.prefillModel = prefillModel
        self.inferRotateModel = nil
        self.prefillRotateModel = nil
    }

    /// 4-function initializer for Gemma3 models with rotation support
    public init(inferModel: MLModel, prefillModel: MLModel, inferRotateModel: MLModel?, prefillRotateModel: MLModel?) {
        self.inferModel = inferModel
        self.prefillModel = prefillModel
        self.inferRotateModel = inferRotateModel
        self.prefillRotateModel = prefillRotateModel
    }

    /// Whether this chunk supports rotation mode (4-function model)
    public var hasRotationSupport: Bool {
        return inferRotateModel != nil && prefillRotateModel != nil
    }
} 