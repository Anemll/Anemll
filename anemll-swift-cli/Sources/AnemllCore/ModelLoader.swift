import Foundation
@preconcurrency import CoreML

/// Model collection for inference.
public struct LoadedModels: @unchecked Sendable {
    public let embedModel: MLModel
    public let lmheadModel: MLModel
    public let ffnChunks: [FFNChunk]
}

/// Protocol for receiving model loading progress updates.
public protocol ModelLoadingProgressDelegate: AnyObject, Sendable {
    /// Called when loading progress changes.
    /// - Parameters:
    ///   - percentage: The overall loading progress from 0.0 to 1.0.
    ///   - stage: Description of the current loading stage.
    ///   - detail: Optional detailed information about the current loading step.
    func loadingProgress(percentage: Double, stage: String, detail: String?)
    
    /// Called when the loading operation has been cancelled.
    func loadingCancelled()
    
    /// Called when all models have been successfully loaded.
    func loadingCompleted(models: LoadedModels)
    
    /// Called when an error occurs during model loading.
    /// - Parameter error: The error that occurred.
    func loadingFailed(error: Error)
}

/// Loads and configures CoreML models with appropriate settings for LLM inference.
public actor ModelLoader {
    /// Configuration for model loading.
    public struct Configuration: Sendable {
        public let computeUnits: MLComputeUnits
        public let allowLowPrecision: Bool
        public let memoryLimit: UInt64?
        public let functionName: String?
        
        public init(
            computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
            //computeUnits: MLComputeUnits = .cpuOnly,
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
    
    /// Progress weights for different loading stages
    private struct ProgressWeights {
        static let embedModel = 0.1
        static let lmheadModel = 0.1
        static let ffnChunks = 0.8  // This is distributed evenly across all chunks
    }
    
    /// The delegate that receives progress updates.
    private weak var progressDelegate: (any ModelLoadingProgressDelegate)?
    
    /// Task that can be cancelled to interrupt the loading process.
    private var loadingTask: Task<LoadedModels, Error>?
    
    /// Initializes a new ModelLoader with an optional progress delegate.
    /// - Parameter progressDelegate: Delegate that will receive progress updates.
    public init(progressDelegate: (any ModelLoadingProgressDelegate)? = nil) {
        self.progressDelegate = progressDelegate
    }
    
    /// Cancels any ongoing model loading.
    public func cancelLoading() {
        loadingTask?.cancel()
        Task { 
            let delegate = self.progressDelegate
            await MainActor.run {
                delegate?.loadingCancelled()
            }
        }
    }
    
    private static func loadMLModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        try MLModel(contentsOf: url, configuration: configuration)
    }
    
    /// Helper class to avoid data races with currentProgress
    private actor ProgressTracker {
        private var currentProgress = 0.0
        private let delegate: (any ModelLoadingProgressDelegate)?
        
        init(delegate: (any ModelLoadingProgressDelegate)?) {
            self.delegate = delegate
        }
        
        func updateProgress(increment: Double, stage: String, detail: String? = nil) async throws {
            if Task.isCancelled {
                throw ModelError.loadingCancelled
            }
            
            currentProgress += increment
            let percentage = min(currentProgress, 1.0)
            
            if let delegate = delegate {
                await MainActor.run {
                    delegate.loadingProgress(
                        percentage: percentage, 
                        stage: stage,
                        detail: detail
                    )
                }
            }
        }
        
        func getCurrentProgress() -> Double {
            return currentProgress
        }
    }
    
    /// Loads a CoreML model with the specified configuration.
    /// - Parameters:
    ///   - config: YAML configuration containing model paths and settings.
    ///   - configuration: Additional CoreML-specific configuration.
    /// - Returns: A LoadedModels instance containing the embeddings, LM head, and FFN chunks.
    @discardableResult
    public func loadModel(
        from config: YAMLConfig,
        configuration: Configuration = Configuration()
    ) async throws -> LoadedModels {
        // Create a task that can be cancelled
        // We need to capture the config and configuration in a Sendable way
        let configCopy = config
        let configurationCopy = configuration
        let progressTracker = ProgressTracker(delegate: progressDelegate)
        
        loadingTask = Task<LoadedModels, Error> {
            print("\nLoading Models:")
            
            // Configure compute units
            let modelConfig = MLModelConfiguration()
            modelConfig.computeUnits = configurationCopy.computeUnits
            
            // Load embeddings model
            try await progressTracker.updateProgress(
                increment: 0.0,
                stage: "Loading Embeddings Model",
                detail: nil
            )
            
            print("\nLoading Embeddings Model:")
            let embedURL = URL(fileURLWithPath: configCopy.embedPath)
            print("Path: \(embedURL.path)")
            let embedModel = try ModelLoader.loadMLModel(at: embedURL, configuration: modelConfig)
            print("✓ Embeddings model loaded")
            
            try await progressTracker.updateProgress(
                increment: ProgressWeights.embedModel,
                stage: "Embeddings Model Loaded",
                detail: configCopy.embedPath
            )
            
            // Load LM head model
            print("\nLoading LM Head Model:")
            let lmheadURL = URL(fileURLWithPath: configCopy.lmheadPath)
            print("Path: \(lmheadURL.path)")
            let lmheadModel = try ModelLoader.loadMLModel(at: lmheadURL, configuration: modelConfig)
            print("✓ LM Head model loaded")
            
            try await progressTracker.updateProgress(
                increment: ProgressWeights.lmheadModel,
                stage: "LM Head Model Loaded",
                detail: configCopy.lmheadPath
            )
            
            // Load all FFN chunks
            print("\nLoading FFN Chunks:")
            var ffnChunks: [FFNChunk] = []
            
            // Calculate per-chunk progress increment
            let chunkProgressIncrement = ProgressWeights.ffnChunks / Double(configCopy.numChunks * 2)
            
            // Validate model files exist before attempting to load
            let fileManager = FileManager.default
            let modelDir = (configCopy.ffnPath as NSString).deletingLastPathComponent
            print("Model directory: \(modelDir)")
            
            // Verify embeddings model
            if !fileManager.fileExists(atPath: configCopy.embedPath) {
                print("❌ ERROR: Embeddings model not found at path: \(configCopy.embedPath)")
                throw ModelError.failedToLoadModel
            }
            
            // Verify LM head model
            if !fileManager.fileExists(atPath: configCopy.lmheadPath) {
                print("❌ ERROR: LM head model not found at path: \(configCopy.lmheadPath)")
                throw ModelError.failedToLoadModel
            }
            
            // For multi-chunk models, verify at least one chunk exists
            if configCopy.numChunks > 1 {
                var foundAnyChunk = false
                var availableChunks: [Int] = []
                
                // Check all possible chunks
                for i in 1...configCopy.numChunks {
                    // Construct expected chunk path
                    let chunkPath: String
                    if configCopy.ffnPath.contains("_chunk_") {
                        // If base path contains chunk info, replace chunk number
                        chunkPath = configCopy.ffnPath.replacingOccurrences(
                            of: "_chunk_\\d+of",
                            with: "_chunk_\(String(format: "%02d", i))of",
                            options: .regularExpression
                        )
                    } else {
                        // Construct from base model path
                        let directory = (configCopy.ffnPath as NSString).deletingLastPathComponent
                        let filename = (configCopy.ffnPath as NSString).lastPathComponent
                        var baseName = filename
                        if baseName.hasSuffix(".mlmodelc") {
                            baseName = String(baseName.dropLast(9))
                        }
                        chunkPath = "\(directory)/\(baseName)_chunk_\(String(format: "%02d", i))of\(String(format: "%02d", configCopy.numChunks)).mlmodelc"
                    }
                    
                    if fileManager.fileExists(atPath: chunkPath) {
                        foundAnyChunk = true
                        availableChunks.append(i)
                    }
                }
                
                if !foundAnyChunk {
                    print("❌ ERROR: No FFN chunks found for model")
                    if let files = try? fileManager.contentsOfDirectory(atPath: modelDir) {
                        print("Available files in \(modelDir):")
                        for file in files {
                            print("  - \(file)")
                        }
                    }
                    throw ModelError.failedToLoadModel
                }
                
                print("✅ Found \(availableChunks.count) available chunks: \(availableChunks)")
            } else {
                // Single chunk model - verify the FFN file exists
                if !fileManager.fileExists(atPath: configCopy.ffnPath) {
                    print("❌ ERROR: FFN model not found at path: \(configCopy.ffnPath)")
                    if let files = try? fileManager.contentsOfDirectory(atPath: modelDir) {
                        print("Available files in \(modelDir):")
                        for file in files {
                            print("  - \(file)")
                        }
                    }
                    throw ModelError.failedToLoadModel
                }
            }
            
            // Load chunks sequentially to avoid memory pressure
            for i in 1...configCopy.numChunks {
                if Task.isCancelled {
                    throw ModelError.loadingCancelled
                }
                
                // Construct the path for this chunk
                var chunkPath: String
                
                // Check if the original path already contains chunk information
                if configCopy.ffnPath.contains("_chunk_") {
                    // If it's already a chunk path, modify the chunk number
                    chunkPath = configCopy.ffnPath.replacingOccurrences(
                        of: "_chunk_\\d+of",
                        with: "_chunk_\(String(format: "%02d", i))of",
                        options: .regularExpression
                    )
                } else if configCopy.numChunks > 1 {
                    // Multi-chunk model with non-chunked path format
                    let directory = (configCopy.ffnPath as NSString).deletingLastPathComponent
                    let filename = (configCopy.ffnPath as NSString).lastPathComponent
                    
                    // Remove .mlmodelc if present
                    var baseName = filename
                    if baseName.hasSuffix(".mlmodelc") {
                        baseName = String(baseName.dropLast(9))
                    }
                    
                    // Add chunk suffix
                    chunkPath = "\(directory)/\(baseName)_chunk_\(String(format: "%02d", i))of\(String(format: "%02d", configCopy.numChunks)).mlmodelc"
                } else {
                    // Single chunk model
                    chunkPath = configCopy.ffnPath
                }
                
                // Skip this chunk if it doesn't exist
                if !fileManager.fileExists(atPath: chunkPath) {
                    print("⚠️ Chunk \(i) not found at: \(chunkPath) - skipping")
                    continue
                }
                
                print("Loading chunk \(i): \(chunkPath)")
                let ffnURL = URL(fileURLWithPath: chunkPath)
                
                // Load inference model for this chunk
                try await progressTracker.updateProgress(
                    increment: 0.0,
                    stage: "Loading FFN Chunk",
                    detail: "Inference \(i)/\(configCopy.numChunks)"
                )
                
                print("Loading inference chunk \(i): \(chunkPath)")
                modelConfig.functionName = "infer"
                
                // Load inference model and fail immediately if it fails
                var inferModel: MLModel
                do {
                    inferModel = try ModelLoader.loadMLModel(at: ffnURL, configuration: modelConfig)
                    print("✅ Inference chunk \(i) loaded")
                } catch {
                    print("❌ Error loading inference chunk \(i): \(error)")
                    throw ModelError.inferenceError("Failed to load inference chunk \(i): \(String(reflecting: error))")
                }
                
                try await progressTracker.updateProgress(
                    increment: chunkProgressIncrement,
                    stage: "FFN Chunk Loaded",
                    detail: "Inference \(i)/\(configCopy.numChunks)"
                )
                
                // Load prefill model for this chunk
                try await progressTracker.updateProgress(
                    increment: 0.0,
                    stage: "Loading FFN Chunk",
                    detail: "Prefill \(i)/\(configCopy.numChunks)"
                )
                
                print("Loading prefill chunk \(i): \(chunkPath)")
                modelConfig.functionName = "prefill"
                
                // Load prefill model and fail immediately if it fails
                var prefillModel: MLModel
                do {
                    prefillModel = try ModelLoader.loadMLModel(at: ffnURL, configuration: modelConfig)
                    print("✅ Prefill chunk \(i) loaded")
                } catch {
                    print("❌ Error loading prefill chunk \(i): \(error)")
                    throw ModelError.inferenceError("Failed to load prefill chunk \(i): \(String(reflecting: error))")
                }
                
                try await progressTracker.updateProgress(
                    increment: chunkProgressIncrement,
                    stage: "FFN Chunk Loaded",
                    detail: "Prefill \(i)/\(configCopy.numChunks)"
                )
                
                ffnChunks.append(FFNChunk(inferModel: inferModel, prefillModel: prefillModel))
            }
            
            // Verify that we loaded all expected chunks
            if ffnChunks.count != configCopy.numChunks {
                print("❌ ERROR: Not all FFN chunks were loaded. Expected \(configCopy.numChunks), got \(ffnChunks.count)")
                throw ModelError.inferenceError("Failed to load all FFN chunks. Expected \(configCopy.numChunks), got \(ffnChunks.count)")
            } else {
                print("✅ Successfully loaded all \(ffnChunks.count) FFN chunks")
            }
            
            // Final update to ensure we reach 100%
            let currentProgress = await progressTracker.getCurrentProgress()
            try await progressTracker.updateProgress(
                increment: max(0, 1.0 - currentProgress),
                stage: "Loading Complete",
                detail: nil
            )
            
            let loadedModels = LoadedModels(
                embedModel: embedModel,
                lmheadModel: lmheadModel,
                ffnChunks: ffnChunks
            )
            
            let delegate = self.progressDelegate
            if let delegate = delegate {
                await MainActor.run {
                    delegate.loadingCompleted(models: loadedModels)
                }
            }
            
            return loadedModels
        }
        
        do {
            // Copy the task reference to avoid actor-isolated property access in closure
            let task = loadingTask!
            return try await withTaskCancellationHandler {
                try await task.value
            } onCancel: { [task] in
                task.cancel()
            }
        } catch {
            let delegate = self.progressDelegate
            if let delegate = delegate {
                await MainActor.run {
                    delegate.loadingFailed(error: error)
                }
            }
            throw error
        }
    }
    
    /// Backward compatibility for static loading without progress reporting
    public static func loadModel(
        from config: YAMLConfig,
        configuration: Configuration = Configuration()
    ) async throws -> LoadedModels {
        // Since YAMLConfig is now Sendable, we can pass it directly to the actor method
        let loader = ModelLoader()
        return try await loader.loadModel(from: config, configuration: configuration)
    }
}

public enum ModelError: Error, Sendable, LocalizedError {
    case failedToLoadModel
    case invalidModelFormat(String)
    case inferenceError(String)
    case loadingCancelled
    
    public var errorDescription: String? {
        switch self {
        case .failedToLoadModel:
            return "Failed to load model"
        case .invalidModelFormat(let message):
            return message
        case .inferenceError(let message):
            return message
        case .loadingCancelled:
            return "Model loading cancelled"
        }
    }
}
