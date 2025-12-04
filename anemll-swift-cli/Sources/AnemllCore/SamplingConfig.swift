import Foundation

/// Configuration for advanced sampling parameters
public struct SamplingConfig: Sendable {
    public let doSample: Bool
    public let temperature: Double
    public let topK: Int
    public let topP: Double
    public let repetitionPenalty: Double
    
    /// Default configuration for greedy decoding (used by regular CLI)
    public static let greedy = SamplingConfig(
        doSample: false,
        temperature: 0.0,
        topK: 0,
        topP: 1.0,
        repetitionPenalty: 1.0
    )
    
    /// Default configuration for sampling (used by advanced CLI)
    public static let defaultSampling = SamplingConfig(
        doSample: true,
        temperature: 0.7,
        topK: 50,
        topP: 0.95,
        repetitionPenalty: 1.1
    )
    
    public init(doSample: Bool, temperature: Double, topK: Int, topP: Double, repetitionPenalty: Double) {
        self.doSample = doSample
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
    }
}