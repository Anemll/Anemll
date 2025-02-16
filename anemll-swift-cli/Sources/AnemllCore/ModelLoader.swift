import Foundation
import CoreML

/// Model collection for inference.
public struct LoadedModels {
    let embedModel: MLModel
    let lmheadModel: MLModel
    let ffnChunks: [FFNChunk]
}

/// Loads and configures CoreML models with appropriate settings for LLM inference.
public class ModelLoader {
    /// Configuration for model loading.
    public struct Configuration {
        public let computeUnits: MLComputeUnits
        public let allowLowPrecision: Bool
        public let memoryLimit: UInt64?
        public let functionName: String?
        
        public init(
            computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
            allowLowPrecision: Bool = false,
            memoryLimit: UInt64? = nil,
            functionName: String? = nil
        ) {
            self.computeUnits = computeUnits
            self.allowLowPrecision = allowLowPrecision
            self.memoryLimit = memoryLimit
            self.functionName = functionName
        }
    }
    
    /// Loads a CoreML model with the specified configuration.
    /// - Parameters:
    ///   - config: YAML configuration containing model paths and settings.
    ///   - configuration: Additional CoreML-specific configuration.
    /// - Returns: A LoadedModels instance containing the embeddings, LM head, and FFN chunks.
    public static func loadModel(
        from config: YAMLConfig,
        configuration: Configuration = Configuration()
    ) throws -> LoadedModels {
        print("\nLoading Models:")
        
        // Configure compute units.
        let modelConfig = MLModelConfiguration()

        modelConfig.computeUnits = MLComputeUnits.cpuAndNeuralEngine
        //configuration.computeUnits
        
        // Load embeddings model.
        print("\nLoading Embeddings Model:")
        let embedURL = URL(fileURLWithPath: config.embedPath)
        print("Path: \(embedURL.path)")
        let embedModel = try MLModel(contentsOf: embedURL, configuration: modelConfig)
        print("✓ Embeddings model loaded")
        
        // Load LM head model.
        print("\nLoading LM Head Model:")
        let lmheadURL = URL(fileURLWithPath: config.lmheadPath)
        print("Path: \(lmheadURL.path)")
        let lmheadModel = try MLModel(contentsOf: lmheadURL, configuration: modelConfig)
        print("✓ LM Head model loaded")
        
        // Load all FFN chunks.
        print("\nLoading FFN Chunks:")
        var ffnChunks: [FFNChunk] = []
        for i in 1...config.numChunks {
            // Modify the FFN path to point to the correct chunk.
            let chunkPath = config.ffnPath.replacingOccurrences(
                of: "01of\(String(format: "%02d", config.numChunks))",
                with: "\(String(format: "%02d", i))of\(String(format: "%02d", config.numChunks))"
            )
            
            let ffnURL = URL(fileURLWithPath: chunkPath)
            
            // Load inference model for this chunk.
            print("Loading inference chunk \(i): \(chunkPath)")
            modelConfig.functionName = "infer"
            let inferModel = try MLModel(contentsOf: ffnURL, configuration: modelConfig)
            print("✓ Inference chunk \(i) loaded")
            
            // Load prefill model for this chunk.
            print("Loading prefill chunk \(i): \(chunkPath)")
            modelConfig.functionName = "prefill"
            let prefillModel = try MLModel(contentsOf: ffnURL, configuration: modelConfig)
            print("✓ Prefill chunk \(i) loaded")
            
            // Create an FFNChunk and add it to the array.
            let chunk = FFNChunk(inferModel: inferModel, prefillModel: prefillModel)
            ffnChunks.append(chunk)
        }
        
        return LoadedModels(
            embedModel: embedModel,
            lmheadModel: lmheadModel,
            ffnChunks: ffnChunks
        )
    }
}

public enum ModelError: Error {
    case failedToLoadModel
    case invalidModelFormat(String)
    case inferenceError(String)
}
