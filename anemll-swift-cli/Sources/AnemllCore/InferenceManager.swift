import Foundation
import CoreVideo
@preconcurrency import CoreML
import CoreFoundation
import Dispatch
import Metal
import IOSurface
import Accelerate

#if arch(x86_64)
// Fallback for builds that don't support native Float16 paths on Intel.
private typealias Float16 = Float
#endif

/// Manages inference by wrapping a CoreML model and handling state.
@preconcurrency public final class InferenceManager: @unchecked Sendable {
    private var hidden_states: Int = -1
    private var embedModel: MLModel!
    private var lmheadModel: MLModel!
    private var ffnChunks: [FFNChunk]!  // Use the FFNChunk defined in FFNChunk.swift
    private var state: MLState!
    private let contextLength: Int
    private let batchSize: Int
    private var fullCausalMask: MLMultiArray?  // Optional - not needed for monolithic argmax models
    private var debugLevel: Int
    private var v110: Bool = false // old conversio has batch x hidden_states for the last chunk
    // Change timing property to CFAbsoluteTime
    private var prefillEndTime: CFAbsoluteTime?
    private var FilterLLAMA01: Bool = false
    private let splitLMHead: Int
    private let modelPrefix: String
    private let vocabSize: Int?
    private let lmHeadChunkSizes: [Int]?
    private let isMonolithic: Bool  // Monolithic model support
    private let argmaxInModel: Bool  // If true, model outputs argmax_idx/val pairs instead of logits
    private let slidingWindow: Int?  // Gemma3 sliding window - if set, use rotation functions when position >= slidingWindow
    private var hasUpdateMask: Bool = false  // If true, prefill model expects update_mask input for KV cache writes
    private var prefillDynamicSlice: Bool = false  // If true, model supports dynamic slice prefill (alternative batch prefill)

    // Pre-allocated update_mask for batch prefill (infer doesn't use update_mask)
    // Shape: [1, 1, contextLength, batchSize] - marks which positions to write for each token in batch
    private var prefillUpdateMask: MLMultiArray?
    private var prefillUpdateMaskBuffer: CVPixelBuffer?

    private var lmheadOutputBackings: [String: MLMultiArray]?
    private var hiddenStatesBackings_emb: [String: MLMultiArray]?  // For embed output
    private var hiddenStatesBackings_ffn: [String: MLMultiArray]?  // For FFN input/output
    private var hiddenStatesBackings_ffnPingPong: [[String: MLMultiArray]] = []  // Alternating FFN outputs for infer
    private var hiddenStatesBackings_last: [String: MLMultiArray]?  // Prefill the last chunk
    private var hiddenStatesBackings_lastPingPong: [[String: MLMultiArray]] = []  // Alternating last-chunk outputs for prefill
    private var hiddenStatesBackings_emb_prefill: [String: MLMultiArray]?  // For embed output in prefill
    private var hiddenStatesBackings_ffn_prefill: [String: MLMultiArray]?  // For FFN output in prefill
    private var hiddenStatesBackings_ffn_prefillPingPong: [[String: MLMultiArray]] = []  // Alternating FFN outputs for prefill

    // Ring buffer for monolithic models to avoid ANE race conditions
    // Using N=16 depth to ensure buffer isn't reused while still being read
    private var monolithicOutputBackingsRing: [[String: MLMultiArray]] = []
    private let monolithicRingBufferDepth = 16
    private var monolithicTokenCounter: Int = 0

    // For argmax mode: store raw pixel buffers separately (to avoid locking issues)
    // MLMultiArray created from pixel buffer may lock it, so we keep raw buffers for direct access
    private var argmaxIdxPixelBuffers: [CVPixelBuffer] = []
    private var argmaxValPixelBuffers: [CVPixelBuffer] = []
    private var GreedySearch = true
    nonisolated(unsafe) private var abort_generation = Int(0)
    private var _busy = false
    private var busy: Bool {
        get { _busy }
        set { _busy = newValue }
    }
    
    // Sampling configuration (defaults to greedy for backward compatibility)
    private var samplingConfig: SamplingConfig = .greedy
    private var generatedTokenHistory: [Int] = []
    private var disablePrefill: Bool = false
    private var disableIOBackings: Bool = false
    private var debugRepeatInferCount: Int = 0
    private var debugRepeatOnlyDivergence: Bool = false
    private var debugCompareKVStateEveryToken: Bool = true
    private var debugPredictReadDelayMs: Double = 0.0

    // Debug: store hidden states for divergence analysis
    private var debugCapturedEmbeddings: MLMultiArray?
    private var debugCapturedFinalHidden: MLMultiArray?
    
    // Debug: KV cache state comparison
    // Since MLState is opaque, we verify state consistency by re-running inference
    private var debugSavedState: MLState?

    private let hiddenStateSimilarityThreshold: Float = 0.9999
    private let kvStateSimilarityThreshold: Float = 0.99999
    private let maxDebugPredictReadDelayMs: Double = 500.0

    // Metal-based argmax for GPU processing (avoids CPU/ANE sync issues)
    private var metalArgmax: MetalArgmax?

    // Serial queue for ANE predictions to ensure thread safety
    // ANE + MLState may not be thread-safe when accessed from different threads
    private let predictionQueue = DispatchQueue(label: "com.anemll.prediction", qos: .userInitiated)

    // Pre-allocated input tensors for sync argmax inference (avoid allocation overhead)
    // All use IOSurface-backed buffers for proper ANE synchronization
    private var argmaxTokenArray: MLMultiArray?
    private var argmaxTokenBuffer: CVPixelBuffer?
    private var argmaxPositionIds: MLMultiArray?
    private var argmaxPositionBuffer: CVPixelBuffer?
    private var argmaxCurrentPosArray: MLMultiArray?
    private var argmaxCurrentPosBuffer: CVPixelBuffer?
    private var argmaxCausalMask: MLMultiArray?
    private var argmaxCausalMaskBuffer: CVPixelBuffer?
    private var lastArgmaxPosition: Int = -1
    private var argmaxInferOptions: MLPredictionOptions?
    private var argmaxInferInput: MLDictionaryFeatureProvider?
    private var cachedArgmaxLayoutByChunks: [Int: (sizes: [Int], offsets: [Int])] = [:]
    
    // Move struct definition to class scope, before the methods
    private struct PartialMax {
        let value: Float
        let index: Int
    }

    private struct FloatBuffer {
        let count: Int
        let float16Ptr: UnsafePointer<Float16>?
        let float32Ptr: UnsafePointer<Float>?
        let unlock: (() -> Void)?
    }

    private func getFloatBuffer(from array: MLMultiArray) throws -> FloatBuffer {
        if let pixelBuffer = array.pixelBuffer {
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
                throw InferenceError.inferenceError("Could not get base address for pixel buffer")
            }
            return FloatBuffer(
                count: array.count,
                float16Ptr: baseAddress.assumingMemoryBound(to: Float16.self),
                float32Ptr: nil,
                unlock: { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
            )
        }

        switch array.dataType {
        case .float16:
            return FloatBuffer(
                count: array.count,
                float16Ptr: array.dataPointer.assumingMemoryBound(to: Float16.self),
                float32Ptr: nil,
                unlock: nil
            )
        case .float32:
            return FloatBuffer(
                count: array.count,
                float16Ptr: nil,
                float32Ptr: array.dataPointer.assumingMemoryBound(to: Float.self),
                unlock: nil
            )
        default:
            throw InferenceError.inferenceError("Unsupported logits data type: \(array.dataType)")
        }
    }

    private func resolveArgmaxChunkLayout(
        numChunks: Int,
        idxArray: MLMultiArray? = nil
    ) -> (sizes: [Int], offsets: [Int]) {
        let safeNumChunks = max(numChunks, 1)

        if let cached = cachedArgmaxLayoutByChunks[safeNumChunks] {
            return cached
        }

        if let configured = lmHeadChunkSizes,
           configured.count == safeNumChunks,
           configured.allSatisfy({ $0 > 0 }) {
            var offsets: [Int] = []
            offsets.reserveCapacity(safeNumChunks)
            var running = 0
            for size in configured {
                offsets.append(running)
                running += size
            }
            let resolved = (configured, offsets)
            cachedArgmaxLayoutByChunks[safeNumChunks] = resolved
            return resolved
        }

        if let vocabSize, vocabSize > 0 {
            let base = vocabSize / safeNumChunks
            let rem = vocabSize % safeNumChunks
            var sizes: [Int] = []
            var offsets: [Int] = []
            sizes.reserveCapacity(safeNumChunks)
            offsets.reserveCapacity(safeNumChunks)
            var running = 0
            for i in 0..<safeNumChunks {
                let size = base + (i < rem ? 1 : 0)
                sizes.append(max(size, 1))
                offsets.append(running)
                running += max(size, 1)
            }
            let resolved = (sizes, offsets)
            cachedArgmaxLayoutByChunks[safeNumChunks] = resolved
            return resolved
        }

        let prefix = modelPrefix.lowercased()
        if safeNumChunks > 0 {
            let fallbackVocab: Int?
            if prefix.hasPrefix("gemma") {
                fallbackVocab = 262144
            } else if prefix.hasPrefix("qwen") {
                fallbackVocab = 151936
            } else if prefix.hasPrefix("llama") {
                fallbackVocab = 128257
            } else {
                fallbackVocab = nil
            }
            if let fallbackVocab {
                let base = fallbackVocab / safeNumChunks
                let rem = fallbackVocab % safeNumChunks
                var sizes: [Int] = []
                var offsets: [Int] = []
                sizes.reserveCapacity(safeNumChunks)
                offsets.reserveCapacity(safeNumChunks)
                var running = 0
                for i in 0..<safeNumChunks {
                    let size = base + (i < rem ? 1 : 0)
                    sizes.append(max(size, 1))
                    offsets.append(running)
                    running += max(size, 1)
                }
                let resolved = (sizes, offsets)
                cachedArgmaxLayoutByChunks[safeNumChunks] = resolved
                return resolved
            }
        }

        var inferredSize = 1
        if let idxArray {
            let n = min(safeNumChunks, idxArray.count)
            for i in 0..<n {
                inferredSize = max(inferredSize, idxArray[i].intValue + 1)
            }
        }

        if inferredSize <= 1 {
            // Legacy fallback (Gemma-style split) if we couldn't infer anything.
            if safeNumChunks == 16 {
                inferredSize = 16384
            } else if safeNumChunks == 8 {
                inferredSize = 32768
            }
        }

        let sizes = Array(repeating: inferredSize, count: safeNumChunks)
        let offsets = (0..<safeNumChunks).map { $0 * inferredSize }
        let resolved = (sizes, offsets)
        cachedArgmaxLayoutByChunks[safeNumChunks] = resolved
        return resolved
    }
    
    public func AbortGeneration( Code : Int)
    {
        abort_generation = Code
    }
    
    public func set_FilterLLAMA01(value: Bool)
    {
        FilterLLAMA01 = value
    }
    
    /// Set sampling configuration for text generation
    public func setSamplingConfig(_ config: SamplingConfig) {
        self.samplingConfig = config
        self.GreedySearch = !config.doSample
    }

    /// Debug option: disable prefill and generate using only the last token.
    public func setDisablePrefill(_ value: Bool) {
        self.disablePrefill = value
    }

    /// Debug option: repeat next-token inference N times (2-4) to detect divergence.
    public func setDebugRepeatInferCount(_ value: Int) {
        if value < 2 {
            self.debugRepeatInferCount = 0
        } else {
            self.debugRepeatInferCount = min(value, 4)
        }
    }

    /// Debug option: when true, suppress per-run repeat logs and only emit divergence.
    public func setDebugRepeatOnlyDivergence(_ value: Bool) {
        self.debugRepeatOnlyDivergence = value
    }

    /// Debug option: compare KV cache snapshots on every repeated token, not just token divergence.
    public func setDebugCompareKVStateEveryToken(_ value: Bool) {
        self.debugCompareKVStateEveryToken = value
    }

    /// Debug option: delay between prediction completion and output read (0...500ms, fractional allowed).
    public func setDebugPredictReadDelayMs(_ value: Int) {
        setDebugPredictReadDelayMs(Double(value))
    }

    /// Debug option: delay between prediction completion and output read (0...500ms, fractional allowed).
    public func setDebugPredictReadDelayMs(_ value: Double) {
        if value.isFinite {
            self.debugPredictReadDelayMs = min(max(value, 0.0), maxDebugPredictReadDelayMs)
        } else {
            self.debugPredictReadDelayMs = 0.0
        }
    }

    private func maybeDelayBeforeReadingPredictionOutputsSync() {
        guard debugPredictReadDelayMs > 0 else { return }
        Thread.sleep(forTimeInterval: debugPredictReadDelayMs / 1000.0)
    }

    private func maybeDelayBeforeReadingPredictionOutputs() async {
        guard debugPredictReadDelayMs > 0 else { return }
        let delayNs = UInt64((debugPredictReadDelayMs * 1_000_000.0).rounded())
        try? await Task.sleep(nanoseconds: delayNs)
    }

    private var ioSurfacePixelBufferAttributes: [String: Any] {
        [
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
    }

    private func runStatefulPredictionOnQueue(
        model: MLModel,
        input: MLFeatureProvider,
        options: MLPredictionOptions
    ) throws {
        var predictionError: Error?
        predictionQueue.sync { [self] in
            do {
                _ = try model.prediction(from: input, using: state, options: options)
            } catch {
                predictionError = error
            }
        }
        if let error = predictionError {
            throw error
        }
    }

    private var monolithicHasRotationSupport: Bool {
        ffnChunks.first?.hasRotationSupport ?? false
    }

    private func monolithicInferModel(for position: Int) throws -> MLModel {
        guard let chunk = ffnChunks?.first else {
            throw InferenceError.inferenceError("Monolithic chunk is not initialized")
        }
        guard let slidingWindow = slidingWindow, position >= slidingWindow else {
            return chunk.inferModel
        }
        if let rotateModel = chunk.inferRotateModel {
            return rotateModel
        }
        throw InferenceError.inferenceError(
            "Position \(position) reached sliding_window=\(slidingWindow), but infer_rotate is unavailable"
        )
    }

    private func monolithicPrefillModel(for batchPos: Int) throws -> MLModel {
        guard let chunk = ffnChunks?.first else {
            throw InferenceError.inferenceError("Monolithic chunk is not initialized")
        }
        guard let slidingWindow = slidingWindow, batchPos >= slidingWindow else {
            return chunk.prefillModel
        }
        if let rotateModel = chunk.prefillRotateModel {
            return rotateModel
        }
        throw InferenceError.inferenceError(
            "Batch position \(batchPos) reached sliding_window=\(slidingWindow), but prefill_rotate is unavailable"
        )
    }

    private func cosineFromAccumulators(dot: Float, normA: Float, normB: Float) -> Float {
        let denominator = sqrt(normA) * sqrt(normB)
        if denominator == 0 {
            return 0.0
        }
        return dot / denominator
    }

    private func bytesPerElement(for dataType: MLMultiArrayDataType) -> Int {
        switch dataType {
        case .float16:
            return MemoryLayout<Float16>.size
        case .float32:
            return MemoryLayout<Float>.size
        case .double:
            return MemoryLayout<Double>.size
        case .int32:
            return MemoryLayout<Int32>.size
        case .int8:
            return MemoryLayout<Int8>.size
        @unknown default:
            return MemoryLayout<Float16>.size
        }
    }

    /// Compute cosine similarity between two MLMultiArrays (for hidden state comparison)
    private func cosineSimilarity(_ a: MLMultiArray, _ b: MLMultiArray) -> Float {
        guard a.count == b.count else {
            print("⚠️ Array size mismatch: \(a.count) vs \(b.count)")
            return -1.0
        }

        let count = a.count
        var dotProduct: Float = 0.0
        var normA: Float = 0.0
        var normB: Float = 0.0

        if a.dataType == .float16, b.dataType == .float16 {
            let ptrA = a.dataPointer.assumingMemoryBound(to: Float16.self)
            let ptrB = b.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count {
                let valA = Float(ptrA[i])
                let valB = Float(ptrB[i])
                dotProduct += valA * valB
                normA += valA * valA
                normB += valB * valB
            }
            return cosineFromAccumulators(dot: dotProduct, normA: normA, normB: normB)
        }

        if a.dataType == .float32, b.dataType == .float32 {
            let ptrA = a.dataPointer.assumingMemoryBound(to: Float.self)
            let ptrB = b.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<count {
                let valA = ptrA[i]
                let valB = ptrB[i]
                dotProduct += valA * valB
                normA += valA * valA
                normB += valB * valB
            }
            return cosineFromAccumulators(dot: dotProduct, normA: normA, normB: normB)
        }

        for i in 0..<count {
            let valA = a[i].floatValue
            let valB = b[i].floatValue
            dotProduct += valA * valB
            normA += valA * valA
            normB += valB * valB
        }

        return cosineFromAccumulators(dot: dotProduct, normA: normA, normB: normB)
    }

    /// Copy MLMultiArray for comparison (deep copy)
    private func copyMLMultiArray(_ source: MLMultiArray) -> MLMultiArray? {
        do {
            let copy = try MLMultiArray(shape: source.shape, dataType: source.dataType)
            let byteCount = source.count * bytesPerElement(for: source.dataType)

            if let pixelBuffer = source.pixelBuffer {
                CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
                defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

                guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                    print("Failed to get base address for pixel buffer-backed MLMultiArray")
                    return nil
                }

                let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
                let width = CVPixelBufferGetWidth(pixelBuffer)
                let height = CVPixelBufferGetHeight(pixelBuffer)
                let elementBytes = bytesPerElement(for: source.dataType)
                let logicalRowBytes = width * elementBytes
                let expectedDataBytes = logicalRowBytes * height

                let dstRaw = copy.dataPointer
                if rowBytes == logicalRowBytes && expectedDataBytes == byteCount {
                    dstRaw.copyMemory(from: baseAddress, byteCount: byteCount)
                } else {
                    // Handle row padding in CVPixelBuffer safely.
                    let srcU8 = baseAddress.assumingMemoryBound(to: UInt8.self)
                    let dstU8 = dstRaw.assumingMemoryBound(to: UInt8.self)
                    let rows = min(height, max(1, byteCount / max(1, logicalRowBytes)))
                    for row in 0..<rows {
                        let srcRow = UnsafeRawPointer(srcU8.advanced(by: row * rowBytes))
                        let dstRow = UnsafeMutableRawPointer(dstU8.advanced(by: row * logicalRowBytes))
                        dstRow.copyMemory(from: srcRow, byteCount: min(logicalRowBytes, byteCount - (row * logicalRowBytes)))
                    }
                }
            } else {
                let srcPtr = source.dataPointer
                let dstPtr = copy.dataPointer
                dstPtr.copyMemory(from: srcPtr, byteCount: byteCount)
            }
            return copy
        } catch {
            print("Failed to copy MLMultiArray: \(error)")
            return nil
        }
    }
    
    /// Get state buffer names from the model
    private func getStateBufferNames() -> [String] {
        guard let ffnChunks = ffnChunks, !ffnChunks.isEmpty else {
            return []
        }
        
        // For non-monolithic models, check the prefillModel state description
        let model = isMonolithic ? ffnChunks[0].inferModel : ffnChunks[0].prefillModel
        
        // Get state description - this contains buffer names
        let stateDesc = model.modelDescription.stateDescriptionsByName
        return Array(stateDesc.keys).sorted()
    }
    
    /// Capture a snapshot of all KV cache buffers from the current state
    /// Returns a dictionary of buffer_name -> copied MLMultiArray
    private func captureKVCacheSnapshot() -> [String: MLMultiArray] {
        guard let state = state else {
            return [:]
        }
        
        let bufferNames = getStateBufferNames()
        var snapshot: [String: MLMultiArray] = [:]

        // Read state buffers on the same serial lane as stateful predictions.
        predictionQueue.sync {
            for name in bufferNames {
                let copied = state.withMultiArray(for: name) { array -> MLMultiArray? in
                    copyMLMultiArray(array)
                }

                if let copied = copied {
                    snapshot[name] = copied
                }
            }
        }
        
        return snapshot
    }
    
    /// Compare two KV cache snapshots and return detailed similarity metrics
    private func compareKVCacheSnapshots(_ snapshot1: [String: MLMultiArray], _ snapshot2: [String: MLMultiArray]) -> (overallSimilarity: Float, perBufferSimilarity: [String: Float], divergentBuffers: [String]) {
        
        var similarities: [Float] = []
        var perBuffer: [String: Float] = [:]
        var divergent: [String] = []
        
        let allKeys = Set(snapshot1.keys).union(snapshot2.keys)
        
        for key in allKeys.sorted() {
            guard let array1 = snapshot1[key], let array2 = snapshot2[key] else {
                // Buffer missing in one snapshot
                perBuffer[key] = -1.0
                divergent.append(key)
                continue
            }
            
            let similarity = cosineSimilarity(array1, array2)
            perBuffer[key] = similarity
            similarities.append(similarity)
            
            if similarity < kvStateSimilarityThreshold {
                divergent.append(key)
            }
        }
        
        let overall = similarities.isEmpty ? 1.0 : similarities.reduce(0, +) / Float(similarities.count)
        
        return (overall, perBuffer, divergent)
    }

    private struct KVBlockSimilarity {
        let blockIndex: Int
        let fullSimilarity: Float
        let focusTokenSimilarity: Float?
        let maxAbsDiff: Float
        let otherTokenMaxAbsDiff: Float?
    }

    private func isRowMajorContiguous(shape: [Int], strides: [Int]) -> Bool {
        guard shape.count == strides.count, !shape.isEmpty else { return false }
        var expectedStride = 1
        for axis in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[axis] != expectedStride {
                return false
            }
            expectedStride *= shape[axis]
        }
        return true
    }

    private func blockLabel(for blockIndex: Int, totalBlocks: Int) -> String {
        if totalBlocks > 1, totalBlocks % 2 == 0 {
            let layer = blockIndex / 2
            let slot = blockIndex % 2
            return "block \(blockIndex) (layer \(layer), slot \(slot))"
        }
        return "block \(blockIndex)"
    }

    private func analyzeKVBufferBlocks(_ before: MLMultiArray, _ after: MLMultiArray, focusTokenIndex: Int?) -> (shape: [Int], blockMetrics: [KVBlockSimilarity])? {
        let shapeA = before.shape.map { $0.intValue }
        let shapeB = after.shape.map { $0.intValue }

        guard before.count == after.count,
              shapeA == shapeB,
              before.dataType == .float16,
              after.dataType == .float16 else {
            return nil
        }

        let shape = shapeA
        let strides = before.strides.map { $0.intValue }

        guard !shape.isEmpty, isRowMajorContiguous(shape: shape, strides: strides) else {
            return nil
        }

        let ptrA = before.dataPointer.assumingMemoryBound(to: Float16.self)
        let ptrB = after.dataPointer.assumingMemoryBound(to: Float16.self)

        let blockCount = max(1, shape[0])
        let blockStride = strides[0]
        let blockElementCount = max(1, shape.dropFirst().reduce(1, *))
        var metrics: [KVBlockSimilarity] = []
        metrics.reserveCapacity(blockCount)

        for block in 0..<blockCount {
            let blockBase = block * blockStride

            var dot: Float = 0
            var normA: Float = 0
            var normB: Float = 0
            var maxAbsDiff: Float = 0

            for idx in 0..<blockElementCount {
                let aVal = Float(ptrA[blockBase + idx])
                let bVal = Float(ptrB[blockBase + idx])
                dot += aVal * bVal
                normA += aVal * aVal
                normB += bVal * bVal
                maxAbsDiff = max(maxAbsDiff, abs(aVal - bVal))
            }

            var focusTokenSimilarity: Float? = nil
            var otherTokenMaxAbsDiff: Float? = nil

            if let tokenIndex = focusTokenIndex {
                if shape.count == 4, tokenIndex >= 0, tokenIndex < shape[2] {
                    let headCount = shape[1]
                    let tokenStride = strides[2]
                    let headStride = strides[1]
                    let dimCount = shape[3]
                    let dimStride = strides[3]

                    var tokenDot: Float = 0
                    var tokenNormA: Float = 0
                    var tokenNormB: Float = 0
                    var nonTokenMaxDiff: Float = 0

                    for head in 0..<headCount {
                        for tok in 0..<shape[2] {
                            for dim in 0..<dimCount {
                                let offset = blockBase + head * headStride + tok * tokenStride + dim * dimStride
                                let aVal = Float(ptrA[offset])
                                let bVal = Float(ptrB[offset])
                                let absDiff = abs(aVal - bVal)
                                if tok == tokenIndex {
                                    tokenDot += aVal * bVal
                                    tokenNormA += aVal * aVal
                                    tokenNormB += bVal * bVal
                                } else {
                                    nonTokenMaxDiff = max(nonTokenMaxDiff, absDiff)
                                }
                            }
                        }
                    }
                    focusTokenSimilarity = cosineFromAccumulators(dot: tokenDot, normA: tokenNormA, normB: tokenNormB)
                    otherTokenMaxAbsDiff = nonTokenMaxDiff
                } else if shape.count == 3, tokenIndex >= 0, tokenIndex < shape[1] {
                    let tokenStride = strides[1]
                    let dimCount = shape[2]
                    let dimStride = strides[2]

                    var tokenDot: Float = 0
                    var tokenNormA: Float = 0
                    var tokenNormB: Float = 0
                    var nonTokenMaxDiff: Float = 0

                    for tok in 0..<shape[1] {
                        for dim in 0..<dimCount {
                            let offset = blockBase + tok * tokenStride + dim * dimStride
                            let aVal = Float(ptrA[offset])
                            let bVal = Float(ptrB[offset])
                            let absDiff = abs(aVal - bVal)
                            if tok == tokenIndex {
                                tokenDot += aVal * bVal
                                tokenNormA += aVal * aVal
                                tokenNormB += bVal * bVal
                            } else {
                                nonTokenMaxDiff = max(nonTokenMaxDiff, absDiff)
                            }
                        }
                    }
                    focusTokenSimilarity = cosineFromAccumulators(dot: tokenDot, normA: tokenNormA, normB: tokenNormB)
                    otherTokenMaxAbsDiff = nonTokenMaxDiff
                }
            }

            metrics.append(
                KVBlockSimilarity(
                    blockIndex: block,
                    fullSimilarity: cosineFromAccumulators(dot: dot, normA: normA, normB: normB),
                    focusTokenSimilarity: focusTokenSimilarity,
                    maxAbsDiff: maxAbsDiff,
                    otherTokenMaxAbsDiff: otherTokenMaxAbsDiff
                )
            )
        }

        return (shape, metrics)
    }

    private func printKVBlockDivergenceAnalysis(
        previousSnapshot: [String: MLMultiArray],
        currentSnapshot: [String: MLMultiArray],
        divergentBuffers: [String],
        focusTokenIndex: Int
    ) {
        guard !divergentBuffers.isEmpty else {
            return
        }

        print("\n  Block-by-block KV cache analysis (focus token index \(focusTokenIndex)):")
        for bufferName in divergentBuffers.prefix(4) {
            guard let before = previousSnapshot[bufferName], let after = currentSnapshot[bufferName] else {
                continue
            }

            guard let analysis = analyzeKVBufferBlocks(before, after, focusTokenIndex: focusTokenIndex) else {
                let shape = before.shape.map { $0.intValue }
                print("    \(bufferName): unsupported detailed analysis (dtype=\(before.dataType), shape=\(shape))")
                continue
            }

            let changedBlocks = analysis.blockMetrics.filter { $0.fullSimilarity < kvStateSimilarityThreshold }
            let totalBlocks = analysis.blockMetrics.count

            if changedBlocks.isEmpty {
                print("    \(bufferName): no per-block divergence despite buffer-level mismatch")
                continue
            }

            print("    \(bufferName) shape=\(analysis.shape): \(changedBlocks.count)/\(totalBlocks) blocks diverged")

            let worstBlocks = changedBlocks.sorted { $0.fullSimilarity < $1.fullSimilarity }.prefix(6)
            for block in worstBlocks {
                let label = blockLabel(for: block.blockIndex, totalBlocks: totalBlocks)
                var line = "      \(label): full=\(String(format: "%.8f", block.fullSimilarity)), maxAbsDiff=\(String(format: "%.6f", block.maxAbsDiff))"
                if let focusSimilarity = block.focusTokenSimilarity {
                    line += ", focusTok=\(String(format: "%.8f", focusSimilarity))"
                }
                if let otherDiff = block.otherTokenMaxAbsDiff {
                    line += ", otherTokMaxAbsDiff=\(String(format: "%.6f", otherDiff))"
                }
                print(line)
            }

            let eps: Float = 1e-5
            let changedOnlyAtFocus = changedBlocks.allSatisfy { ($0.otherTokenMaxAbsDiff ?? 0.0) < eps }
            if changedOnlyAtFocus, changedBlocks.contains(where: { ($0.focusTokenSimilarity ?? 1.0) < kvStateSimilarityThreshold }) {
                print("      -> changes are concentrated at focus token \(focusTokenIndex)")
            }
        }
    }


    public init(models: LoadedModels, contextLength: Int, batchSize: Int, splitLMHead: Int = 8, debugLevel: Int = 0, v110: Bool = false, argmaxInModel: Bool = false, slidingWindow: Int? = nil, updateMaskPrefill: Bool = false, prefillDynamicSlice: Bool = false, disableIOBackings: Bool = false, modelPrefix: String = "llama", vocabSize: Int? = nil, lmHeadChunkSizes: [Int]? = nil) throws {  // Make init throwing
#if arch(x86_64)
        throw InferenceError.inferenceError("x86_64 has no Apple Neural Engine and is unsupported by this application.")
#endif
        self.debugLevel = debugLevel
        self.isMonolithic = models.isMonolithic
        self.argmaxInModel = argmaxInModel
        self.slidingWindow = slidingWindow
        self.embedModel = models.embedModel
        self.lmheadModel = models.lmheadModel
        // Assume models.ffnChunks is available (see note below)
        self.ffnChunks = models.ffnChunks
        self.contextLength = contextLength
        self.batchSize = batchSize
        self.splitLMHead = splitLMHead
        self.modelPrefix = modelPrefix
        self.v110 = v110 // Set the v110 flag based on the parameter
        self.disableIOBackings = disableIOBackings
        self.vocabSize = vocabSize
        self.lmHeadChunkSizes = lmHeadChunkSizes

        // Check if rotation functions are available
        let hasRotation = ffnChunks.first?.hasRotationSupport ?? false

        // Use update_mask_prefill from config, or fallback to model input detection
        if updateMaskPrefill {
            hasUpdateMask = true
        } else if isMonolithic, let prefillModel = ffnChunks.first?.prefillModel {
            // Fallback: detect from model inputs (for backward compatibility)
            hasUpdateMask = prefillModel.modelDescription.inputDescriptionsByName["update_mask"] != nil
        }

        // Set prefill dynamic slice flag
        self.prefillDynamicSlice = prefillDynamicSlice

        // Match Python tests/chat.py behavior:
        // allow partial-batch prefill if either update_mask_prefill or
        // prefill_dynamic_slice is available.
        let allowBatchPrefill = hasUpdateMask || prefillDynamicSlice

        print("InferenceManager initialized with v110=\(v110), splitLMHead=\(splitLMHead), batchSize=\(batchSize), isMonolithic=\(isMonolithic), argmaxInModel=\(argmaxInModel), slidingWindow=\(slidingWindow != nil ? "\(slidingWindow!)" : "nil"), hasRotation=\(hasRotation), hasUpdateMask=\(hasUpdateMask), prefillDynamicSlice=\(prefillDynamicSlice), allowBatchPrefill=\(allowBatchPrefill), disableIOBackings=\(disableIOBackings)")

        // Print prefill mode info (matching Python chat_full.py)
        if allowBatchPrefill {
            print("✅ Batch prefill enabled (partial batches supported)")
        } else {
            print("⚠️  No update_mask_prefill or prefill_dynamic_slice; partial batches use single-token prefill")
        }

        if isMonolithic, let slidingWindow, !hasRotation {
            print("⚠️  Monolithic model has sliding_window=\(slidingWindow) but rotate functions are unavailable; generation will stop before that boundary to avoid ANE failure")
        }

        // Create full causal mask - needed for attention
        self.fullCausalMask = try MLMultiArray(shape: [1, 1, NSNumber(value: contextLength), NSNumber(value: contextLength)], dataType: .float16)
        initFullCausalMask()

        self.initState()

        try initializeBackings()

        // Pre-allocate input tensors for sync argmax inference (eliminates allocation overhead)
        // int32 arrays use regular MLMultiArray (pixel buffer only supports fp16/uint8)
        // fp16 causal mask uses IOSurface-backed pixel buffer for ANE synchronization
        if argmaxInModel && isMonolithic {
            // int32 input arrays - regular MLMultiArray
            argmaxTokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
            argmaxPositionIds = try MLMultiArray(shape: [1], dataType: .int32)
            argmaxCurrentPosArray = try MLMultiArray(shape: [1], dataType: .int32)

            // Causal mask [1, 1, 1, contextLength]
            if disableIOBackings {
                argmaxCausalMaskBuffer = nil
                argmaxCausalMask = try MLMultiArray(
                    shape: [1, 1, 1, NSNumber(value: contextLength)],
                    dataType: .float16
                )
                // Initialize causal mask with -inf
                let ptr = argmaxCausalMask!.dataPointer.assumingMemoryBound(to: Float16.self)
                for i in 0..<contextLength {
                    ptr[i] = Float16(-Float.infinity)
                }
            } else {
                // IOSurface-backed fp16 pixel buffer for ANE synchronization
                let ioAttributes: [String: Any] = [
                    kCVPixelBufferMetalCompatibilityKey as String: true,
                    kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
                ]

                var maskBuffer: CVPixelBuffer?
                let maskStatus = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    contextLength, 1,
                    kCVPixelFormatType_OneComponent16Half,
                    ioAttributes as CFDictionary,
                    &maskBuffer
                )
                guard maskStatus == kCVReturnSuccess, let mBuf = maskBuffer else {
                    throw InferenceError.inferenceError("Failed to create causal mask pixel buffer")
                }
                argmaxCausalMaskBuffer = mBuf
                argmaxCausalMask = MLMultiArray(pixelBuffer: mBuf, shape: [1, 1, 1, NSNumber(value: contextLength)])

                // Initialize causal mask with -inf
                CVPixelBufferLockBaseAddress(mBuf, [])
                if let baseAddress = CVPixelBufferGetBaseAddress(mBuf) {
                    let ptr = baseAddress.assumingMemoryBound(to: Float16.self)
                    for i in 0..<contextLength {
                        ptr[i] = Float16(-Float.infinity)
                    }
                }
                CVPixelBufferUnlockBaseAddress(mBuf, [])
            }
            lastArgmaxPosition = -1

            // Pre-allocate input feature provider (reused for all inferences)
            argmaxInferInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": argmaxTokenArray!,
                "position_ids": argmaxPositionIds!,
                "causal_mask": argmaxCausalMask!,
                "current_pos": argmaxCurrentPosArray!
            ])

            // Pre-allocate prediction options
            argmaxInferOptions = MLPredictionOptions()
        }

        // Pre-allocate update_mask for prefill if model supports it
        // Shape: [1, 1, contextLength, batchSize] - each column marks the write position for that token in batch
        if hasUpdateMask && isMonolithic {
            if disableIOBackings {
                prefillUpdateMaskBuffer = nil
                prefillUpdateMask = try MLMultiArray(
                    shape: [1, 1, NSNumber(value: contextLength), NSNumber(value: batchSize)],
                    dataType: .float16
                )
                // Initialize to zeros
                let totalElements = contextLength * batchSize
                let ptr = prefillUpdateMask!.dataPointer.assumingMemoryBound(to: Float16.self)
                for i in 0..<totalElements {
                    ptr[i] = Float16(0.0)
                }
            } else {
                let ioAttributes: [String: Any] = [
                    kCVPixelBufferMetalCompatibilityKey as String: true,
                    kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
                ]

                // Create pixel buffer: width = batchSize, height = contextLength
                var updateMaskBuffer: CVPixelBuffer?
                let updateMaskStatus = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    batchSize, contextLength,
                    kCVPixelFormatType_OneComponent16Half,
                    ioAttributes as CFDictionary,
                    &updateMaskBuffer
                )
                guard updateMaskStatus == kCVReturnSuccess, let uBuf = updateMaskBuffer else {
                    throw InferenceError.inferenceError("Failed to create update_mask pixel buffer")
                }
                prefillUpdateMaskBuffer = uBuf
                prefillUpdateMask = MLMultiArray(pixelBuffer: uBuf, shape: [1, 1, NSNumber(value: contextLength), NSNumber(value: batchSize)])

                // Initialize to zeros
                CVPixelBufferLockBaseAddress(uBuf, [])
                if let baseAddress = CVPixelBufferGetBaseAddress(uBuf) {
                    let totalElements = contextLength * batchSize
                    let ptr = baseAddress.assumingMemoryBound(to: Float16.self)
                    for i in 0..<totalElements {
                        ptr[i] = Float16(0.0)
                    }
                }
                CVPixelBufferUnlockBaseAddress(uBuf, [])
            }

            print("Pre-allocated update_mask buffer: [\(contextLength), \(batchSize)]")
        }

        // Metal argmax is available but CPU Accelerate SIMD is faster for this workload
        // Metal overhead (IOSurface locking, command buffer) outweighs GPU benefit
        // Keeping Metal code for reference but using CPU by default
        if isMonolithic && debugLevel >= 2 {
            self.metalArgmax = MetalArgmax()
            if metalArgmax != nil {
                print("Metal argmax available (disabled by default, CPU is faster)")
            }
        }

        // Debug model descriptions
        if debugLevel >= 1 && !isMonolithic {
            print("\nLM Head Model Output Description:")
            if let lmhead = lmheadModel {
                for (name, desc) in lmhead.modelDescription.outputDescriptionsByName {
                    print("Output \(name):")
                    print("- Type: \(type(of: desc.type))")
                    print("- Description: \(desc.type)")
                }
            }
        }

        // Debug monolithic model descriptions
        if debugLevel >= 1 && isMonolithic {
            print("\nMonolithic Model Output Description:")
            for (name, desc) in ffnChunks[0].inferModel.modelDescription.outputDescriptionsByName {
                print("Output \(name):")
                print("- Type: \(type(of: desc.type))")
                print("- Description: \(desc.type)")
            }
        }
    }
    
    public func initializeBackings() throws {
        if isMonolithic {
            // For monolithic models, initialize logits output backings from the monolithic model
            try initializeMonolithicOutputBackings()
        } else {
            // Initialize output backings for lmhead
            try initializeLMHeadOutputBackings()

            // Initialize hidden states backings
            try initializeHiddenStatesBackings()

            try initializePrefillBackings()
            try initializeLastChunkBacking()
        }
    }
    
    
    public func initFullCausalMask()  {
        // Create full causal mask once with -inf and 0.0
        // Optimized using direct pointer access for large context lengths
        guard let mask = fullCausalMask else {
            print("Skipping initFullCausalMask - mask is nil")
            return
        }

        let totalCount = mask.count
        let startTime = CFAbsoluteTimeGetCurrent()

        // Use direct pointer access for Float16 - MUCH faster than NSNumber subscripting
        // Shape is [1, 1, contextLength, contextLength], stored row-major
        let ptr = mask.dataPointer.assumingMemoryBound(to: Float16.self)

        // Fill entire array with -inf first (fast memset-like operation)
        let negInf = Float16(-Float.infinity)
        for i in 0..<totalCount {
            ptr[i] = negInf
        }

        // Set causal pattern: for row i, columns 0..i should be 0.0 (visible)
        // Index in flat array for [0, 0, i, j] = i * contextLength + j
        let zero = Float16(0.0)
        for i in 0..<contextLength {
            let rowOffset = i * contextLength
            // Set columns 0 through i to 0.0
            for j in 0...(i) {
                ptr[rowOffset + j] = zero
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("initFullCausalMask completed in \(String(format: "%.3f", elapsed))s for context \(contextLength)")
    }
    
    public func initState()  {
        // For monolithic models, create state from inferModel (like Python does)
        // This ensures state compatibility when switching between prefill and infer functions
        if isMonolithic {
            self.state = ffnChunks[0].inferModel.makeState()
        } else {
            self.state = ffnChunks[0].prefillModel.makeState()
        }
    }

    private func resetStateForPrefill() {
        guard let chunks = ffnChunks, !chunks.isEmpty else {
            return
        }
        if isMonolithic {
            state = chunks[0].inferModel.makeState()
        } else {
            state = chunks[0].prefillModel.makeState()
        }
        lastArgmaxPosition = -1
    }

    private func rebuildStateForContext(
        _ tokens: [Int],
        contextPos: Int,
        tokenizer: Tokenizer
    ) async throws -> Int {
        if tokens.isEmpty || contextPos <= 0 {
            return 0
        }
        let safePos = min(contextPos, tokens.count)
        if disablePrefill {
            var pos = 0
            while pos < safePos {
                let _ = try await generateNextToken(
                    for: tokens[pos],
                    currentPos: pos + 1,
                    temperature: 0,
                    tokenizer: tokenizer
                )
                pos += 1
            }
            return safePos
        } else {
            var tmpTokens = tokens
            return try await runPrefill(on: &tmpTokens, contextPos: safePos, tokenizer: tokenizer)
        }
    }
    
    public func ToggeDebugLevel()  {
        if (debugLevel == 0 ) {
            debugLevel = 2
        }else{
            debugLevel = 0
        }
        print("Debug level set to \(debugLevel)")
    }
    
    private func initializeLMHeadOutputBackings() throws {
        let outputDescription = lmheadModel.modelDescription.outputDescriptionsByName
        var outputBackingsDict: [String: MLMultiArray] = [:]

        // For argmax mode: LM head outputs argmax_idx and argmax_val instead of logits
        if argmaxInModel {
            print("Initializing LM head output backings for argmax mode (non-monolithic)")

            // Create argmax_idx backing (int32) - model outputs int32 indices
            let idxArray = try MLMultiArray(shape: [NSNumber(value: splitLMHead)], dataType: .int32)
            outputBackingsDict["argmax_idx"] = idxArray

            // Create argmax_val backing (fp16) - model outputs fp16 values
            let valArray = try MLMultiArray(shape: [NSNumber(value: splitLMHead)], dataType: .float16)
            outputBackingsDict["argmax_val"] = valArray

            lmheadOutputBackings = outputBackingsDict
            return
        }

        // Standard logits mode: LM head outputs logits1..logitsN
        let featureNames = (1...splitLMHead).map { i in "logits\(i)" }

        for featureName in featureNames {
            guard let featureDesc = outputDescription[featureName] else {
                throw InferenceError.inferenceError("Missing feature description for \(featureName)")
            }

            if debugLevel >= 1 {
                print("\nFeature \(featureName) type: \(featureDesc.type)")
            }

            // Check if it's a multiarray feature and get its constraint
            guard featureDesc.type.rawValue == 5,
                  let constraint = featureDesc.multiArrayConstraint else {
                print("Feature \(featureName) type details:")
                print("- Type: \(type(of: featureDesc.type))")
                print("- Description: \(featureDesc.type)")
                throw InferenceError.inferenceError("Feature \(featureName) is not a multiarray")
            }

            let shape = constraint.shape

            if disableIOBackings {
                // Use standard MLMultiArray output backings (no CVPixelBuffer)
                let outputBacking = try MLMultiArray(shape: shape, dataType: constraint.dataType)
                outputBackingsDict[featureName] = outputBacking
                continue
            }

            // Calculate dimensions for pixel buffer
            let lastDim = shape.last?.intValue ?? 1
            let otherDims = shape.dropLast().reduce(1) { $0 * $1.intValue }

            // Create IOSurface-backed pixel buffer
            let attributes = ioSurfacePixelBufferAttributes

            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(
                kCFAllocatorDefault,
                lastDim,     // Width is last dimension
                otherDims,   // Height is product of other dimensions
                kCVPixelFormatType_OneComponent16Half,
                attributes as CFDictionary,
                &pixelBuffer
            )
            if debugLevel >= 2 {
                print("Creating pixel buffer for \(featureName):")
                print("- Width (last dim): \(lastDim)")
                print("- Height (other dims): \(otherDims)")
                print("- Status: \(status)")
            }
            guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
                throw InferenceError.inferenceError("Failed to create pixel buffer for \(featureName)")
            }

            // Create MLMultiArray from pixel buffer
            let outputBacking = MLMultiArray(pixelBuffer: buffer, shape: shape)
            outputBackingsDict[featureName] = outputBacking
        }

        lmheadOutputBackings = outputBackingsDict
    }
    
    private func initializeHiddenStatesBackings() throws {
        // Check embedding model shapes first
        if debugLevel >= 1 {
            print("\n=== Embedding Model Shapes ===")
            for (name, desc) in embedModel.modelDescription.inputDescriptionsByName {
                if let constraint = desc.multiArrayConstraint {
                    print("Embed Input \(name):", constraint.shape.map { $0.intValue })
                }
            }
            for (name, desc) in embedModel.modelDescription.outputDescriptionsByName {
                if let constraint = desc.multiArrayConstraint {
                    print("Embed Output \(name):", constraint.shape.map { $0.intValue })
                }
            }
        }
        
        // Get shape from FFN model's input
        if let desc = ffnChunks[0].inferModel.modelDescription.inputDescriptionsByName["hidden_states"],
           let constraint = desc.multiArrayConstraint {
            let shape = constraint.shape
            
            if debugLevel >= 1 {
                print("\n=== FFN Model Shapes ===")
                print("FFN Model Input Shape:", shape.map { $0.intValue })
                print("\nFFN Model Features:")
                print("Inputs:", ffnChunks[0].inferModel.modelDescription.inputDescriptionsByName.keys)
                print("Outputs:", ffnChunks[0].inferModel.modelDescription.outputDescriptionsByName.keys)
            }
            
            let lastDim = shape.last?.intValue ?? 2048
            self.hidden_states = lastDim
            let otherDims = shape.dropLast().reduce(1) { $0 * $1.intValue }
            hiddenStatesBackings_ffnPingPong.removeAll(keepingCapacity: true)

            if disableIOBackings {
                let dataType = constraint.dataType
                hiddenStatesBackings_emb = ["hidden_states": try MLMultiArray(shape: shape, dataType: dataType)]
                hiddenStatesBackings_ffnPingPong = [
                    ["output_hidden_states": try MLMultiArray(shape: shape, dataType: dataType)],
                    ["output_hidden_states": try MLMultiArray(shape: shape, dataType: dataType)]
                ]
                hiddenStatesBackings_ffn = hiddenStatesBackings_ffnPingPong.first

                if debugLevel >= 1 {
                    print("Single-token embed backing shape:", shape.map { $0.intValue })
                    print("Single-token FFN ping-pong buffers: \(hiddenStatesBackings_ffnPingPong.count)")
                }
                return
            }
            
            let attributes = ioSurfacePixelBufferAttributes
            
            // Create embed output backing
            var embedPixelBuffer: CVPixelBuffer?
            let embedStatus = CVPixelBufferCreate(
                kCFAllocatorDefault,
                lastDim,
                otherDims,
                kCVPixelFormatType_OneComponent16Half,
                attributes as CFDictionary,
                &embedPixelBuffer
            )
            
            guard embedStatus == kCVReturnSuccess, let embedBuffer = embedPixelBuffer else {
                throw InferenceError.inferenceError("Failed to create pixel buffer for embed output")
            }
            
            // Store embed output backing
            hiddenStatesBackings_emb = ["hidden_states": MLMultiArray(pixelBuffer: embedBuffer, shape: shape)]
            
            if debugLevel >= 1 {
                print("Single-token embed backing shape:", shape.map { $0.intValue })
            }
            
            // Create two FFN output backings and alternate them to avoid read/write reuse hazards.
            for slot in 0..<2 {
                var ffnPixelBuffer: CVPixelBuffer?
                let ffnStatus = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    lastDim,
                    otherDims,
                    kCVPixelFormatType_OneComponent16Half,
                    attributes as CFDictionary,
                    &ffnPixelBuffer
                )

                guard ffnStatus == kCVReturnSuccess, let ffnBuffer = ffnPixelBuffer else {
                    throw InferenceError.inferenceError("Failed to create pixel buffer for FFN output slot \(slot)")
                }

                hiddenStatesBackings_ffnPingPong.append(["output_hidden_states": MLMultiArray(pixelBuffer: ffnBuffer, shape: shape)])
            }

            hiddenStatesBackings_ffn = hiddenStatesBackings_ffnPingPong.first
            if debugLevel >= 1 {
                print("Single-token FFN ping-pong buffers: \(hiddenStatesBackings_ffnPingPong.count)")
            }
        }
    }


    private func initializeLastChunkBacking() throws {
        guard let desc = ffnChunks.last?.prefillModel.modelDescription.outputDescriptionsByName["output_hidden_states"],
            let constraint = desc.multiArrayConstraint else {
            throw InferenceError.inferenceError("Failed to get last chunk output description")
        }
        
        let hiddenSize = constraint.shape.last?.intValue ?? self.hidden_states
    
        let shape: [NSNumber] = [1, 1, NSNumber(value: hiddenSize)]
        hiddenStatesBackings_lastPingPong.removeAll(keepingCapacity: true)

        if disableIOBackings {
            hiddenStatesBackings_lastPingPong = [
                ["output_hidden_states": try MLMultiArray(shape: shape, dataType: constraint.dataType)],
                ["output_hidden_states": try MLMultiArray(shape: shape, dataType: constraint.dataType)]
            ]
            hiddenStatesBackings_last = hiddenStatesBackings_lastPingPong.first

            if debugLevel >= 1 {
                print("\nLast Chunk Backing Initialized:")
                print("Shape: \(shape.map { $0.intValue })")
                print("Last-chunk ping-pong buffers: \(hiddenStatesBackings_lastPingPong.count)")
            }
            return
        }
        
        for slot in 0..<2 {
            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(
                kCFAllocatorDefault,
                hiddenSize,  // width
                1,           // height (batch=1)
                kCVPixelFormatType_OneComponent16Half,
                ioSurfacePixelBufferAttributes as CFDictionary,
                &pixelBuffer
            )

            guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
                throw InferenceError.inferenceError("Failed to create last chunk pixel buffer slot \(slot): \(status)")
            }

            hiddenStatesBackings_lastPingPong.append(["output_hidden_states": MLMultiArray(pixelBuffer: buffer, shape: shape)])
        }
        hiddenStatesBackings_last = hiddenStatesBackings_lastPingPong.first
        
        if debugLevel >= 1 {
            print("\nLast Chunk Backing Initialized:")
            print("Shape: \(shape.map { $0.intValue })")
            print("Last-chunk ping-pong buffers: \(hiddenStatesBackings_lastPingPong.count)")
        }
    }
    
    private func initializePrefillBackings() throws {
        let hiddenSize = self.hidden_states  // Adjust based on your model's hidden size
        let shape: [NSNumber] = [1, NSNumber(value: batchSize), NSNumber(value: hiddenSize)]
        let attributes = ioSurfacePixelBufferAttributes
        hiddenStatesBackings_ffn_prefillPingPong.removeAll(keepingCapacity: true)

        if debugLevel >= 1 {
            print("\n=== Prefill Backing Initialization ===")
            print("Hidden size:", hiddenSize)
            print("Batch size:", batchSize)
            print("Prefill backing shape:", shape.map { $0.intValue })
        }

        if disableIOBackings {
            hiddenStatesBackings_emb_prefill = ["hidden_states": try MLMultiArray(shape: shape, dataType: .float16)]
            hiddenStatesBackings_ffn_prefillPingPong = [
                ["output_hidden_states": try MLMultiArray(shape: shape, dataType: .float16)],
                ["output_hidden_states": try MLMultiArray(shape: shape, dataType: .float16)]
            ]
            hiddenStatesBackings_ffn_prefill = hiddenStatesBackings_ffn_prefillPingPong.first

            if debugLevel >= 1 {
                print("Embed prefill backing created with shape:", shape.map { $0.intValue })
                print("FFN prefill ping-pong buffers: \(hiddenStatesBackings_ffn_prefillPingPong.count)")
            }
            return
        }

        // Embedding prefill backing
        var embedPixelBuffer: CVPixelBuffer?
        let embedStatus = CVPixelBufferCreate(
            kCFAllocatorDefault,
            hiddenSize,  // Width
            batchSize,   // Height
            kCVPixelFormatType_OneComponent16Half,
            attributes as CFDictionary,
            &embedPixelBuffer
        )
        guard embedStatus == kCVReturnSuccess, let embedBuffer = embedPixelBuffer else {
            throw InferenceError.inferenceError("Failed to create embed prefill pixel buffer")
        }
        hiddenStatesBackings_emb_prefill = ["hidden_states": MLMultiArray(pixelBuffer: embedBuffer, shape: shape)]

        if debugLevel >= 1 {
            print("Embed prefill backing created with shape:", shape.map { $0.intValue })
        }

        // FFN prefill backing ping-pong
        for slot in 0..<2 {
            var ffnPixelBuffer: CVPixelBuffer?
            let ffnStatus = CVPixelBufferCreate(
                kCFAllocatorDefault,
                hiddenSize,
                batchSize,
                kCVPixelFormatType_OneComponent16Half,
                attributes as CFDictionary,
                &ffnPixelBuffer
            )
            guard ffnStatus == kCVReturnSuccess, let ffnBuffer = ffnPixelBuffer else {
                throw InferenceError.inferenceError("Failed to create FFN prefill pixel buffer slot \(slot)")
            }
            hiddenStatesBackings_ffn_prefillPingPong.append(["output_hidden_states": MLMultiArray(pixelBuffer: ffnBuffer, shape: shape)])
        }
        hiddenStatesBackings_ffn_prefill = hiddenStatesBackings_ffn_prefillPingPong.first
        if debugLevel >= 1 {
            print("FFN prefill ping-pong buffers: \(hiddenStatesBackings_ffn_prefillPingPong.count)")
        }
    }

    private func initializeMonolithicOutputBackings() throws {
        // Ring buffer with N=16 depth to avoid ANE race conditions.
        // This ensures buffers aren't reused while still being read/written.
        let outputDescription = ffnChunks[0].inferModel.modelDescription.outputDescriptionsByName

        if debugLevel >= 1 {
            print("\n=== Initializing Monolithic Output Backings (Ring Buffer N=\(monolithicRingBufferDepth)) ===")
            print("Available outputs: \(outputDescription.keys)")
            print("ArgmaxInModel: \(argmaxInModel)")
        }

        // For argmax mode: use regular MLMultiArray output backings (NOT pixel buffers)
        // Pixel buffers only support Float16/UInt8, but argmax_idx may be int32
        // The arrays are small (16 elements) so overhead is negligible
        if argmaxInModel {
            let numChunks = splitLMHead  // 16 for 262K vocab

            monolithicOutputBackingsRing = []
            for bufferIndex in 0..<monolithicRingBufferDepth {
                var outputBackingsDict: [String: MLMultiArray] = [:]

                // Create argmax_idx backing (int32) - model outputs int32 indices
                let idxArray = try MLMultiArray(shape: [NSNumber(value: numChunks)], dataType: .int32)
                outputBackingsDict["argmax_idx"] = idxArray

                // Create argmax_val backing (fp16) - model outputs fp16 values
                let valArray = try MLMultiArray(shape: [NSNumber(value: numChunks)], dataType: .float16)
                outputBackingsDict["argmax_val"] = valArray

                monolithicOutputBackingsRing.append(outputBackingsDict)
            }

            monolithicTokenCounter = 0
            print("✅ Argmax mode: using MLMultiArray backings with ring buffer (depth=\(monolithicRingBufferDepth))")
            return
        }

        // For logits mode, use pixel buffer backings for efficient large array access
        let featureNames = (1...splitLMHead).map { i in "logits\(i)" }

        // Create N ring buffer slots
        monolithicOutputBackingsRing = []

        for bufferIndex in 0..<monolithicRingBufferDepth {
            var outputBackingsDict: [String: MLMultiArray] = [:]

            for featureName in featureNames {
                guard let featureDesc = outputDescription[featureName] else {
                    if debugLevel >= 1 {
                        print("Warning: Feature \(featureName) not found in monolithic model outputs")
                    }
                    continue
                }

                guard featureDesc.type.rawValue == 5,
                      let constraint = featureDesc.multiArrayConstraint else {
                    print("Feature \(featureName) is not a multiarray")
                    throw InferenceError.inferenceError("Feature \(featureName) is not a multiarray")
                }

                let shape = constraint.shape
                let lastDim = shape.last?.intValue ?? 1
                let otherDims = shape.dropLast().reduce(1) { $0 * $1.intValue }

                if bufferIndex == 0 {
                    if disableIOBackings {
                        print("  \(featureName): shape=\(shape.map { $0.intValue }) (MLMultiArray)")
                    } else {
                        print("  \(featureName): shape=\(shape.map { $0.intValue }), pixelBuffer=\(lastDim)x\(otherDims)")
                    }
                }

                if disableIOBackings {
                    outputBackingsDict[featureName] = try MLMultiArray(shape: shape, dataType: constraint.dataType)
                    continue
                }

                // IOSurface-backed buffer for ANE compatibility and polling
                let attributes: [String: Any] = [
                    kCVPixelBufferMetalCompatibilityKey as String: true,
                    kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
                ]

                // Create CVPixelBuffer with original dimensions
                var pixelBuffer: CVPixelBuffer?
                let status = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    lastDim, otherDims,
                    kCVPixelFormatType_OneComponent16Half,
                    attributes as CFDictionary,
                    &pixelBuffer
                )
                guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
                    throw InferenceError.inferenceError("Failed to create pixel buffer \(bufferIndex) for \(featureName)")
                }
                outputBackingsDict[featureName] = MLMultiArray(pixelBuffer: buffer, shape: shape)
            }

            monolithicOutputBackingsRing.append(outputBackingsDict)

            if debugLevel >= 2 {
                print("Created ring buffer slot \(bufferIndex) with \(outputBackingsDict.count) outputs")
            }
        }

        monolithicTokenCounter = 0
        print("✅ Monolithic output backings initialized with ring buffer (depth=\(monolithicRingBufferDepth)) for \(featureNames.count) logits outputs")
    }
    
    // Helper to get causal mask slice for current position
    // Optimized with direct pointer access for performance
    private func getCausalMask(for length: Int, at position: Int, paddingLength: Int? = nil) throws -> MLMultiArray {
        // Ensure position is within bounds
        let safePosition = min(position, contextLength - 1)

        // Create mask with correct dimensions
        let mask = try MLMultiArray(
            shape: [1, 1, NSNumber(value: length), NSNumber(value: contextLength)],
            dataType: .float16
        )

        // Use direct pointer access for performance
        let ptr = mask.dataPointer.assumingMemoryBound(to: Float16.self)
        let negInf = Float16(-Float.infinity)
        let zero = Float16(0.0)

        // Fill mask with -inf by default
        let totalCount = mask.count
        for i in 0..<totalCount {
            ptr[i] = negInf
        }

        // Set causal attention pattern
        // Shape is [1, 1, length, contextLength], index = i * contextLength + j
        for i in 0..<length {
            let rowOffset = i * contextLength
            let visibleEnd = min(safePosition + i, contextLength - 1)
            for j in 0...visibleEnd {
                ptr[rowOffset + j] = zero
            }
        }

        // Apply padding if specified
        if let paddingLength = paddingLength {
            for i in paddingLength..<length {
                let rowOffset = i * contextLength
                for j in 0..<contextLength {
                    ptr[rowOffset + j] = negInf
                }
            }
        }

        if debugLevel >= 2 {
            print("\nCausal mask for length \(length) at position \(position):")
            print("Shape:", mask.shape.map { $0.intValue })
        }

        return mask
    }
    
    private func debugPrint(_ message: String, level: Int = 1) {
        if debugLevel >= level {
            print(message)
        }
    }
    
    private func debugTokens(_ tokens: [Int], prefix: String, tokenizer: Tokenizer? = nil) {
        if debugLevel >= 1 {
            print("\n\(prefix) tokens: \(tokens)")
            if let tokenizer = tokenizer {
                print("\(prefix) decoded: \(tokenizer.decode(tokens: tokens))")
            }
        }
    }
    
    public func runStPrefill(
        on contextTokens: inout [Int],
        contextPos: Int,
        tokenizer: Tokenizer? = nil
    ) async throws -> Int {
        let inputLength = contextTokens.prefix(contextPos).count
        for i in 0..<inputLength {
            let _ = try await generateNextToken(
                for: contextTokens[i],
                currentPos: i+1,
                temperature: 0,
                tokenizer: tokenizer
            )
            if debugLevel >= 1 {
                print("runStPrefill predicted token:  \(i) \(contextTokens[i])")
            }
        }
        return inputLength
    }

    public func runPrefill(
        on contextTokens: inout [Int],
        contextPos: Int,
        tokenizer: Tokenizer? = nil
    ) async throws -> Int {
        if debugLevel >= 1 {
            print("\n=== Starting Prefill Phase ===")
            print("Input context length:", contextPos)
            print("Configured batch size:", batchSize)
            print("Is monolithic:", isMonolithic)
            debugTokens(Array(contextTokens.prefix(contextPos)), prefix: "Input")
        }
        guard let ffnChunks = ffnChunks else {
            throw InferenceError.inferenceError("ffnChunks was nil in runPrefill()")
        }

        // For monolithic models, use the monolithic prefill path
        if isMonolithic {
            return try await runMonolithicPrefill(on: &contextTokens, contextPos: contextPos, tokenizer: tokenizer)
        }
        var batchPos = 0

        // Process FULL batches only with prefill model (to avoid padding issues that corrupt KV cache)
        while batchPos + batchSize <= contextPos {
            let batchEnd = batchPos + batchSize

            if debugLevel >= 1 {
                print("\nPrefill batch: \(batchPos) to \(batchEnd), full batch of \(batchSize)")
            }
            
            // Create input tensor for current batch (full batch)
            let batchInput = try MLMultiArray(shape: [1, NSNumber(value: batchSize)], dataType: .int32)
            for i in 0..<batchSize {
                batchInput[[0, i] as [NSNumber]] = NSNumber(value: contextTokens[batchPos + i])
            }
            
            // Generate position IDs
            let positionIds = try MLMultiArray(shape: [NSNumber(value: batchSize)], dataType: .int32)
            for i in 0..<batchSize {
                positionIds[i] = NSNumber(value: batchPos + i)
            }
            
            // Create batch causal mask
            let batchCausalMask = try MLMultiArray(
                shape: [1, 1, NSNumber(value: batchSize), NSNumber(value: contextLength)],  // Always use full contextLength
                dataType: .float16
            )
            
            // Fill with -inf by default
            for i in 0..<batchCausalMask.count {
                batchCausalMask[i] = NSNumber(value: Float(-Float.infinity))
            }
            
            // Set causal attention pattern
            for i in 0..<batchSize {
                for j in 0..<contextLength {  // Use full contextLength
                    if j <= (batchPos + i) {
                        batchCausalMask[[0, 0, i, j] as [NSNumber]] = NSNumber(value: Float(0.0))
                    }
                }
            }
            
            // Run embeddings with prefill backing
            let embedInput = try MLDictionaryFeatureProvider(dictionary: ["input_ids": batchInput])
            let embedOptions = MLPredictionOptions()
            if let backings = hiddenStatesBackings_emb_prefill {
                embedOptions.outputBackings = backings
                if debugLevel >= 1 {
                    print("Using embedding prefill backing with shape:", backings["hidden_states"]?.shape.map { $0.intValue } ?? [])
                    print("Embedding input shape:", batchInput.shape.map { $0.intValue })
                }
            }
            
            if debugLevel >= 1 {
                print("About to run embedding model prediction...")
            }
            let _ = try await embedModel.prediction(from: embedInput, options: embedOptions)
            await maybeDelayBeforeReadingPredictionOutputs()
            if debugLevel >= 1 {
                print("Embedding model prediction completed successfully")
            }
            
            guard let hiddenStates = hiddenStatesBackings_emb_prefill?["hidden_states"] else {
                throw InferenceError.inferenceError("Missing embed prefill output backing")
            }
            
            if debugLevel >= 1 {
                print("Retrieved hidden states from embedding with shape:", hiddenStates.shape.map { $0.intValue })
            }
            
            // Process FFN chunks
            var currentHiddenStates = hiddenStates  // Shape: [1, 128, hidden_states]
            let chunkCount = ffnChunks.count
            
            // Determine if we should use rotation mode (for Gemma3 with sliding window)
            let useRotation = slidingWindow != nil && batchPos >= slidingWindow!
            if useRotation && debugLevel >= 1 {
                print("Using prefill rotation mode for batchPos \(batchPos) >= slidingWindow \(slidingWindow!)")
            }

            for (index, chunk) in ffnChunks.enumerated() {
                let isLastChunk = index == chunkCount - 1
                let ffnOptions = MLPredictionOptions()

                if debugLevel >= 1 {
                    print("\nFFN chunk \(index + 1)/\(chunkCount), isLastChunk: \(isLastChunk)")
                    print("Current hidden states shape:", currentHiddenStates.shape.map { $0.intValue })
                }

                // Assign output backing BEFORE predict
                // Check what shape the model expects by looking at its OUTPUT description
                var useLastChunkBacking = false
                var selectedBackings: [String: MLMultiArray]?

                // Use the appropriate model to check output description based on rotation mode
                let modelToCheck = useRotation ? (chunk.prefillRotateModel ?? chunk.prefillModel) : chunk.prefillModel
                if let outputDesc = modelToCheck.modelDescription.outputDescriptionsByName["output_hidden_states"],
                   let constraint = outputDesc.multiArrayConstraint {
                    let expectedBatchDim = constraint.shape[1].intValue
                    if debugLevel >= 1 {
                        print("Chunk \(index + 1) prefill model expects output shape: \(constraint.shape.map { $0.intValue })")
                    }
                    // If model expects output batch dim of 1, use last chunk backing
                    useLastChunkBacking = (expectedBatchDim == 1)
                }

                if useLastChunkBacking && !v110 {
                    let pool = hiddenStatesBackings_lastPingPong.isEmpty
                        ? (hiddenStatesBackings_last.map { [$0] } ?? [])
                        : hiddenStatesBackings_lastPingPong
                    if !pool.isEmpty {
                        let chosenBackings = pool[index % pool.count]
                        selectedBackings = chosenBackings
                        ffnOptions.outputBackings = chosenBackings  // Shape: [1, 1, hidden_states]
                        if debugLevel >= 1 {
                            print("Using last chunk backing slot \(index % pool.count) with shape:", selectedBackings?["output_hidden_states"]?.shape.map { $0.intValue } ?? [])
                        }
                    }
                } else {
                    // For models expecting batch shape or when v110=true
                    let pool = hiddenStatesBackings_ffn_prefillPingPong.isEmpty
                        ? (hiddenStatesBackings_ffn_prefill.map { [$0] } ?? [])
                        : hiddenStatesBackings_ffn_prefillPingPong
                    if !pool.isEmpty {
                        let chosenBackings = pool[index % pool.count]
                        selectedBackings = chosenBackings
                        ffnOptions.outputBackings = chosenBackings  // Shape: [1, batch_size, hidden_states]
                        if debugLevel >= 1 {
                            print("Using FFN prefill backing slot \(index % pool.count) with shape:", selectedBackings?["output_hidden_states"]?.shape.map { $0.intValue } ?? [])
                        }
                    }
                }

                let currentPosArray = try MLMultiArray(shape: [1], dataType: .int32)
                currentPosArray[0] = NSNumber(value: batchPos)

                let prefillInput = try MLDictionaryFeatureProvider(dictionary: [
                    "hidden_states": currentHiddenStates,  // Shape: [1, 128, hidden_states]
                    "position_ids": positionIds,
                    "causal_mask": batchCausalMask,
                    "current_pos": currentPosArray
                ])

                // Use rotation function if available and batchPos >= slidingWindow
                if useRotation, let prefillRotateModel = chunk.prefillRotateModel {
                    try runStatefulPredictionOnQueue(
                        model: prefillRotateModel,
                        input: prefillInput,
                        options: ffnOptions
                    )
                } else {
                    try runStatefulPredictionOnQueue(
                        model: chunk.prefillModel,
                        input: prefillInput,
                        options: ffnOptions
                    )
                }
                await maybeDelayBeforeReadingPredictionOutputs()

                guard let nextHiddenStates = selectedBackings?["output_hidden_states"] else {
                    if useLastChunkBacking && !v110 {
                        throw InferenceError.inferenceError("Missing last chunk output backing")
                    }
                    throw InferenceError.inferenceError("Missing FFN prefill output backing")
                }
                currentHiddenStates = nextHiddenStates

                if debugLevel >= 2 {
                    debugTensor(currentHiddenStates, prefix: "FFN chunk \(index + 1) output")
                }
            }
            
            batchPos = batchEnd
        }

        // Process remaining tokens one-at-a-time using infer model (avoids padding issues that corrupt KV cache)
        if batchPos < contextPos {
            if debugLevel >= 1 {
                print("\nProcessing remaining \(contextPos - batchPos) tokens one-at-a-time with infer model")
            }
            while batchPos < contextPos {
                // Use generateNextToken which processes single token through embed + FFN chunks + lmhead
                // We don't need the returned token, just need to populate KV cache
                let _ = try await generateNextToken(
                    for: contextTokens[batchPos],
                    currentPos: batchPos + 1,  // generateNextToken uses 1-indexed positions
                    temperature: 0,
                    tokenizer: tokenizer
                )
                if debugLevel >= 1 {
                    print("  Prefill single token at pos \(batchPos): \(contextTokens[batchPos])")
                }
                batchPos += 1
            }
        }

        return contextPos
    }

    /// Run prefill for monolithic models - passes input_ids directly to the model
    /// With update_mask: processes all tokens in batches (including partial batches)
    /// Without update_mask: processes full batches with prefill, remaining one-at-a-time with infer
    private func runMonolithicPrefill(
        on contextTokens: inout [Int],
        contextPos: Int,
        tokenizer: Tokenizer? = nil
    ) async throws -> Int {
        if debugLevel >= 1 {
            print("\n=== Running Monolithic Prefill ===")
            print("Context position:", contextPos)
            print("Batch size:", batchSize)
            print("Has update_mask:", hasUpdateMask)
        }

        var batchPos = 0

        // Match Python tests/chat.py monolithic behavior:
        // only update_mask enables partial-batch prefill.
        // Otherwise we use full batches + single-token fallback for remaining tokens.
        let processPartialWithPrefill = hasUpdateMask

        // Process batches with prefill model
        while batchPos < contextPos {
            let remainingTokens = contextPos - batchPos
            let currentBatchSize = processPartialWithPrefill ? min(remainingTokens, batchSize) : batchSize

            // Without update_mask, stop when we don't have a full batch
            if !processPartialWithPrefill && remainingTokens < batchSize {
                break
            }

            if debugLevel >= 1 {
                let batchType = currentBatchSize == batchSize ? "full" : "partial"
                print("\nMonolithic prefill batch: \(batchPos) to \(batchPos + currentBatchSize), \(batchType) batch (\(currentBatchSize) tokens)")
            }

            // Create input tensor for current batch (padded to batchSize)
            let batchInput = try MLMultiArray(shape: [1, NSNumber(value: batchSize)], dataType: .int32)
            for i in 0..<batchSize {
                if i < currentBatchSize {
                    batchInput[[0, i] as [NSNumber]] = NSNumber(value: contextTokens[batchPos + i])
                } else {
                    batchInput[[0, i] as [NSNumber]] = NSNumber(value: 0)  // Pad with zeros
                }
            }

            // Generate position IDs for full batch (padded positions don't matter with update_mask)
            let positionIds = try MLMultiArray(shape: [NSNumber(value: batchSize)], dataType: .int32)
            for i in 0..<batchSize {
                positionIds[i] = NSNumber(value: batchPos + i)
            }

            // Create batch causal mask
            let batchCausalMask = try MLMultiArray(
                shape: [1, 1, NSNumber(value: batchSize), NSNumber(value: contextLength)],
                dataType: .float16
            )

            // Fill with -inf by default
            for i in 0..<batchCausalMask.count {
                batchCausalMask[i] = NSNumber(value: Float(-Float.infinity))
            }

            // Set causal attention pattern
            for i in 0..<batchSize {
                for j in 0..<contextLength {
                    if j <= (batchPos + i) {
                        batchCausalMask[[0, 0, i, j] as [NSNumber]] = NSNumber(value: Float(0.0))
                    }
                }
            }

            // Create current_pos as tensor
            let currentPosArray = try MLMultiArray(shape: [1], dataType: .int32)
            currentPosArray[0] = NSNumber(value: batchPos)

            // Build input dictionary
            var inputDict: [String: MLMultiArray] = [
                "input_ids": batchInput,
                "position_ids": positionIds,
                "causal_mask": batchCausalMask,
                "current_pos": currentPosArray
            ]

            // Add update_mask if model supports it
            if hasUpdateMask, let updateMask = prefillUpdateMask {
                // Populate update_mask: set 1.0 at position [batchPos + i, i] for each valid token
                // Shape is [1, 1, contextLength, batchSize]
                if let updateMaskBuffer = prefillUpdateMaskBuffer {
                    // Pixel buffer path
                    CVPixelBufferLockBaseAddress(updateMaskBuffer, [])
                    if let baseAddress = CVPixelBufferGetBaseAddress(updateMaskBuffer) {
                        let ptr = baseAddress.assumingMemoryBound(to: Float16.self)
                        let rowBytes = CVPixelBufferGetBytesPerRow(updateMaskBuffer)
                        let stride = rowBytes / MemoryLayout<Float16>.size

                        // Clear entire buffer first
                        for row in 0..<contextLength {
                            for col in 0..<batchSize {
                                ptr[row * stride + col] = Float16(0.0)
                            }
                        }

                        // Set 1.0 at the write positions for each token in the batch
                        for i in 0..<currentBatchSize {
                            let writePos = batchPos + i
                            if writePos < contextLength {
                                ptr[writePos * stride + i] = Float16(1.0)
                            }
                        }
                    }
                    CVPixelBufferUnlockBaseAddress(updateMaskBuffer, [])
                } else {
                    // MLMultiArray path
                    let ptr = updateMask.dataPointer.assumingMemoryBound(to: Float16.self)
                    let rowStride = batchSize
                    let totalElements = contextLength * batchSize
                    for i in 0..<totalElements {
                        ptr[i] = Float16(0.0)
                    }
                    for i in 0..<currentBatchSize {
                        let writePos = batchPos + i
                        if writePos < contextLength {
                            ptr[writePos * rowStride + i] = Float16(1.0)
                        }
                    }
                }

                inputDict["update_mask"] = updateMask

                if debugLevel >= 1 {
                    print("  update_mask set for positions \(batchPos) to \(batchPos + currentBatchSize - 1)")
                }
            }

            // Create input feature provider
            let prefillInput = try MLDictionaryFeatureProvider(dictionary: inputDict)

            // Select rotate/non-rotate prefill function based on position.
            let prefillModel = try monolithicPrefillModel(for: batchPos)

            // Run prediction on serial queue for consistent execution context
            var predictionError: Error?
            predictionQueue.sync { [self] in
                do {
                    _ = try prefillModel.prediction(
                        from: prefillInput,
                        using: state,
                        options: MLPredictionOptions()
                    )
                } catch {
                    predictionError = error
                }
            }
            if let error = predictionError {
                throw error
            }

            if debugLevel >= 1 {
                print("✅ Monolithic prefill batch completed")
            }

            batchPos += currentBatchSize
        }

        // Process remaining tokens one-at-a-time using infer model (only needed without update_mask)
        if batchPos < contextPos {
            if debugLevel >= 1 {
                print("\nProcessing \(contextPos - batchPos) remaining tokens one-at-a-time")
            }
        }

        while batchPos < contextPos {
            let token = contextTokens[batchPos]

            // Create single-token input
            let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
            tokenArray[[0, 0] as [NSNumber]] = NSNumber(value: token)

            // Single position ID
            let positionIds = try MLMultiArray(shape: [1], dataType: .int32)
            positionIds[0] = NSNumber(value: batchPos)

            // Single-token causal mask
            let singleMask = try MLMultiArray(
                shape: [1, 1, 1, NSNumber(value: contextLength)],
                dataType: .float16
            )
            for j in 0..<contextLength {
                singleMask[[0, 0, 0, j] as [NSNumber]] = j <= batchPos
                    ? NSNumber(value: Float(0.0))
                    : NSNumber(value: Float(-Float.infinity))
            }

            // Current position
            let currentPosArray = try MLMultiArray(shape: [1], dataType: .int32)
            currentPosArray[0] = NSNumber(value: batchPos)

            let inferInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": tokenArray,
                "position_ids": positionIds,
                "causal_mask": singleMask,
                "current_pos": currentPosArray
            ])

            // Select rotate/non-rotate infer function based on position.
            let inferModel = try monolithicInferModel(for: batchPos)

            var predictionError: Error?
            predictionQueue.sync { [self] in
                do {
                    _ = try inferModel.prediction(
                        from: inferInput,
                        using: state,
                        options: MLPredictionOptions()
                    )
                } catch {
                    predictionError = error
                }
            }
            if let error = predictionError {
                throw error
            }

            batchPos += 1
        }

        // Initialize argmax causal mask for token generation phase
        // After prefill at contextPos, positions 0..contextPos-1 should be visible
        if argmaxInModel {
            if let maskBuffer = argmaxCausalMaskBuffer {
                CVPixelBufferLockBaseAddress(maskBuffer, [])
                if let baseAddress = CVPixelBufferGetBaseAddress(maskBuffer) {
                    let ptr = baseAddress.assumingMemoryBound(to: Float16.self)
                    // Reset mask to -inf and set visible positions
                    for i in 0..<contextLength {
                        ptr[i] = Float16(-Float.infinity)
                    }
                    for j in 0..<min(contextPos, contextLength) {
                        ptr[j] = Float16(0.0)
                    }
                }
                CVPixelBufferUnlockBaseAddress(maskBuffer, [])
                lastArgmaxPosition = contextPos - 1
            } else if let maskArray = argmaxCausalMask {
                let ptr = maskArray.dataPointer.assumingMemoryBound(to: Float16.self)
                for i in 0..<contextLength {
                    ptr[i] = Float16(-Float.infinity)
                }
                for j in 0..<min(contextPos, contextLength) {
                    ptr[j] = Float16(0.0)
                }
                lastArgmaxPosition = contextPos - 1
            }
        }

        return contextPos
    }

    /// Apply repetition penalty to logits based on generated token history
    func applyRepetitionPenalty(logits: inout [Float], penalty: Double) {
        guard penalty != 1.0 && !generatedTokenHistory.isEmpty else { return }
        
        let uniqueTokens = Set(generatedTokenHistory)
        for tokenId in uniqueTokens {
            if tokenId < logits.count {
                if logits[tokenId] < 0 {
                    logits[tokenId] *= Float(penalty)
                } else {
                    logits[tokenId] /= Float(penalty)
                }
            }
        }
    }
    
    /// Extremely fast top-k sampling using heap-like selection
    func topKSample(logits: [Float], temperature: Float, topK: Int) -> Int {
        guard topK > 0 && topK < logits.count else {
            // If topK is 0 or >= vocab size, use all tokens
            return topPSample(logits: logits, temperature: temperature, topP: 1.0)
        }
        
        // Find top-k using quickselect-like algorithm (much faster than full sort)
        var indexedLogits: [(Int, Float)] = []
        indexedLogits.reserveCapacity(logits.count)
        
        for (i, logit) in logits.enumerated() {
            indexedLogits.append((i, logit / temperature))
        }
        
        // Partial sort to get top-k efficiently
        indexedLogits.sort { $0.1 > $1.1 }
        let topK_indices = Array(indexedLogits.prefix(topK))
        
        // Fast softmax on top-k only
        let maxLogit = topK_indices[0].1
        var expSum: Float = 0.0
        var probs: [(Int, Float)] = []
        probs.reserveCapacity(topK)
        
        for (idx, logit) in topK_indices {
            let expVal = exp(logit - maxLogit)
            expSum += expVal
            probs.append((idx, expVal))
        }
        
        // Sample using cumulative distribution
        let r = Float.random(in: 0..<expSum)
        var cumulative: Float = 0.0
        
        for (idx, expVal) in probs {
            cumulative += expVal
            if r <= cumulative {
                return idx
            }
        }
        
        return probs.last!.0
    }
    
    /// Optimized multinomial sampling from logits
    func sampleFromLogits(_ logits: [Float]) -> Int {
        // Find max for numerical stability
        let maxLogit = logits.max() ?? 0
        
        // Compute exp values and sum in one pass
        var expSum: Float = 0.0
        var expValues: [Float] = []
        expValues.reserveCapacity(logits.count)
        
        for logit in logits {
            let expVal = exp(logit - maxLogit)
            expValues.append(expVal)
            expSum += expVal
        }
        
        // Sample using cumulative distribution without normalizing
        let r = Float.random(in: 0..<expSum)
        var cumulative: Float = 0.0
        
        for (idx, expVal) in expValues.enumerated() {
            cumulative += expVal
            if r <= cumulative {
                return idx
            }
        }
        
        return logits.count - 1  // Fallback
    }
    
    func topPSample(logits: [Float], temperature: Float = 1.0, topP: Float = 0.9) -> Int {
        // Early exit for topP = 1.0 (no filtering)
        if topP >= 1.0 {
            return sampleFromLogits(logits.map { $0 / temperature })
        }
        
        // Apply temperature and find max for numerical stability
        let invTemp = 1.0 / temperature
        let maxLogit = logits.max() ?? 0
        
        // Create indexed probabilities in one pass
        var indexedProbs: [(Int, Float)] = []
        indexedProbs.reserveCapacity(logits.count)
        var expSum: Float = 0.0
        
        for (i, logit) in logits.enumerated() {
            let scaledLogit = (logit - maxLogit) * invTemp
            let expVal = exp(scaledLogit)
            expSum += expVal
            indexedProbs.append((i, expVal))
        }
        
        // Sort by probability (descending) and accumulate
        indexedProbs.sort { $0.1 > $1.1 }
        
        var cumulative: Float = 0.0
        var cutoffIndex = indexedProbs.count
        
        for (idx, (_, expVal)) in indexedProbs.enumerated() {
            cumulative += expVal / expSum  // Normalize on the fly
            if cumulative >= topP {
                cutoffIndex = idx + 1
                break
            }
        }
        
        // Sample directly from the filtered set without renormalization
        let filteredSum = indexedProbs.prefix(cutoffIndex).reduce(0) { $0 + $1.1 }
        let r = Float.random(in: 0..<filteredSum)
        
        var acc: Float = 0.0
        for (tokenIdx, expVal) in indexedProbs.prefix(cutoffIndex) {
            acc += expVal
            if r <= acc {
                return tokenIdx
            }
        }
        
        return indexedProbs[0].0  // Fallback to highest prob
    }

    /// Generates the next token given the current token. This method calls the embedding model,
    /// passes the output through each FFN chunk's infer function, and then runs the LM head.
    public func generateNextToken(
        for lastToken: Int,
        currentPos: Int,
        temperature: Float,
        tokenizer: Tokenizer? = nil,
        captureHiddenStates: Bool = false  // Enable hidden state capture for debugging
    ) async throws -> Int {
        guard let ffnChunks = ffnChunks else {
            throw InferenceError.inferenceError("ffnChunks is nil before generateNextToken")
        }
        if debugLevel >= 1 {
            print("\nGenerating token at position \(currentPos-1)")
            print("Input token: \(lastToken)", terminator: "")
            if let tokenizer = tokenizer {
                print(" (\(tokenizer.decode(tokens: [lastToken])))")
            } else {
                print()
            }
        }

        let _padTokenId = tokenizer?.padTokenId ?? 0 // Default to 0 if nil

        // For monolithic models, use the monolithic inference path
        if isMonolithic {
            return try await generateNextTokenMonolithic(
                for: lastToken,
                currentPos: currentPos,
                temperature: temperature,
                tokenizer: tokenizer
            )
        }

        // Run embeddings with output backing
        let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenArray[[0, 0] as [NSNumber]] = NSNumber(value: lastToken)
        let embedInput = try MLDictionaryFeatureProvider(dictionary: ["input_ids": tokenArray])

        // Use embed output backing
        let embedOptions = MLPredictionOptions()
        if let backings = hiddenStatesBackings_emb {
            embedOptions.outputBackings = backings
        }
        let _ = try await embedModel.prediction(from: embedInput, options: embedOptions)
        await maybeDelayBeforeReadingPredictionOutputs()

        // Get hidden states from embed backing
        guard let hiddenStates = hiddenStatesBackings_emb?["hidden_states"] else {
            throw InferenceError.inferenceError("Missing embed output backing")
        }

        // Capture embeddings if requested
        if captureHiddenStates {
            debugCapturedEmbeddings = copyMLMultiArray(hiddenStates)
        }

        // Create position IDs (1D) - use currentPos-1 since currentPos is 1-indexed
        let safePos = currentPos - 1
        let positionIds = try MLMultiArray(shape: [1], dataType: .int32)
        positionIds[0] = NSNumber(value: safePos)

        // Get causal mask for single token - use safePos to match position_ids
        // At position N, we should see positions 0 to N (not 0 to N+1)
        let causalMask = try getCausalMask(for: 1, at: safePos)

        // Create current_pos as tensor
        let currentPosArray = try MLMultiArray(shape: [1], dataType: .int32)
        currentPosArray[0] = NSNumber(value: safePos)

        // Run through FFN chunks using FFN backing
        var currentHiddenStates = hiddenStates

        // Determine if we should use rotation mode (for Gemma3 with sliding window)
        // safePos is the actual 0-indexed position
        let useRotation = slidingWindow != nil && safePos >= slidingWindow!
        if useRotation && debugLevel >= 1 {
            print("Using rotation mode for position \(safePos) >= slidingWindow \(slidingWindow!)")
        }

        let ffnInferPool = hiddenStatesBackings_ffnPingPong.isEmpty
            ? (hiddenStatesBackings_ffn.map { [$0] } ?? [])
            : hiddenStatesBackings_ffnPingPong

        for (chunkIndex, chunk) in ffnChunks.enumerated() {
            let ffnOptions = MLPredictionOptions()
            let selectedBackings = ffnInferPool.isEmpty ? nil : ffnInferPool[chunkIndex % ffnInferPool.count]
            if let backings = selectedBackings {
                ffnOptions.outputBackings = backings
            }

            let inferInput = try MLDictionaryFeatureProvider(dictionary: [
                "hidden_states": currentHiddenStates,
                "position_ids": positionIds,
                "causal_mask": causalMask,
                "current_pos": currentPosArray
            ])

            // Use rotation function if available and position >= slidingWindow
            if useRotation, let inferRotateModel = chunk.inferRotateModel {
                try runStatefulPredictionOnQueue(
                    model: inferRotateModel,
                    input: inferInput,
                    options: ffnOptions
                )
            } else {
                try runStatefulPredictionOnQueue(
                    model: chunk.inferModel,
                    input: inferInput,
                    options: ffnOptions
                )
            }
            await maybeDelayBeforeReadingPredictionOutputs()

            guard let nextHiddenStates = selectedBackings?["output_hidden_states"] ?? hiddenStatesBackings_ffn?["output_hidden_states"] else {
                throw InferenceError.inferenceError("Missing FFN output backing")
            }
            currentHiddenStates = nextHiddenStates
        }

        debugHiddenStates(currentHiddenStates, prefix: "Final hidden states to LM head")
        
        // Capture final hidden states before LM head if requested
        if captureHiddenStates {
            debugCapturedFinalHidden = copyMLMultiArray(currentHiddenStates)
        }

        // Run LM head with final hidden states
        let lmOptions = MLPredictionOptions()
        if let backings = lmheadOutputBackings {
            lmOptions.outputBackings = backings
        }

        let lmInput = try MLDictionaryFeatureProvider(dictionary: ["hidden_states": currentHiddenStates])
        let _ = try await lmheadModel.prediction(from: lmInput, options: lmOptions)
        await maybeDelayBeforeReadingPredictionOutputs()
        
        guard let outputBackings = lmheadOutputBackings else {
            throw InferenceError.inferenceError("Output backings not initialized")
        }

        // For argmax mode (non-monolithic): LM head already computed argmax, just read the results
        if argmaxInModel && !isMonolithic {
            guard let idxArray = outputBackings["argmax_idx"],
                  let valArray = outputBackings["argmax_val"] else {
                throw InferenceError.inferenceError("Missing argmax_idx or argmax_val in LM head output backings")
            }

            let numChunks = idxArray.count
            let layout = resolveArgmaxChunkLayout(numChunks: numChunks, idxArray: idxArray)
            let chunkSizes = layout.sizes
            let chunkOffsets = layout.offsets

            // Collect ALL chunk data for debugging
            var allChunkData: [(chunk: Int, localIdx: Int, val: Float)] = []
            
            // Find the chunk with highest value
            var bestChunk = 0
            var bestLocalIdx = 0
            var bestVal: Float = -Float.infinity
            
            for i in 0..<numChunks {
                let localIdx = idxArray[i].intValue
                let val = Float(valArray[i].floatValue)
                allChunkData.append((chunk: i, localIdx: localIdx, val: val))
                
                if val > bestVal {
                    bestVal = val
                    bestChunk = i
                    bestLocalIdx = localIdx
                }
            }

            // Get the global index
            let globalIdx = bestLocalIdx + chunkOffsets[bestChunk]

            if debugLevel >= 1 {
                // Sort by value to see top candidates
                let sorted = allChunkData.sorted { $0.val > $1.val }
                
                print("\n=== LM Head Argmax Debug (Non-Monolithic) ===")
                print("Position: \(currentPos - 1)")
                print("Top 5 chunks:")
                for i in 0..<min(5, sorted.count) {
                    let c = sorted[i]
                    let gIdx = c.localIdx + chunkOffsets[c.chunk]
                    let marker = (c.chunk == bestChunk) ? " <-- SELECTED" : ""
                    print("  #\(i+1): chunk=\(String(format: "%2d", c.chunk)), localIdx=\(String(format: "%5d", c.localIdx)), globalIdx=\(String(format: "%6d", gIdx)), offset=\(String(format: "%6d", chunkOffsets[c.chunk])), size=\(String(format: "%5d", chunkSizes[c.chunk])), val=\(String(format: "%.8f", c.val))\(marker)")
                    
                    // Decode token if we have tokenizer
                    if let tokenizer = tokenizer {
                        let decoded = tokenizer.decode(tokens: [gIdx], skipSpecialTokens: false)
                        print("       decoded: \"\(decoded)\"")
                    }
                }
                
                // Check if top values are very close (precision issue indicator)
                if sorted.count >= 2 {
                    let diff = abs(sorted[0].val - sorted[1].val)
                    print("\nValue comparison:")
                    print("  top-1 val: \(String(format: "%.8f", sorted[0].val))")
                    print("  top-2 val: \(String(format: "%.8f", sorted[1].val))")
                    print("  difference: \(String(format: "%.8f", diff))")
                    
                    if diff < 0.001 {
                        print("  ⚠️ WARNING: Values are very close - potential precision/race issue!")
                    }
                    if diff < 0.0001 {
                        print("  🔴 CRITICAL: Values differ by less than 0.0001 - likely non-determinism!")
                    }
                }
                
                // Platform info
                #if os(iOS)
                print("\nPlatform: iOS (potential A-series ANE issue)")
                #elseif os(macOS)
                print("\nPlatform: macOS (M-series)")
                #else
                print("\nPlatform: Other")
                #endif
                
                print("Result: chunk=\(bestChunk), local_idx=\(bestLocalIdx), global_idx=\(globalIdx), val=\(String(format: "%.8f", bestVal))")
                print("=========================================\n")
            }

            return globalIdx
        }

        // Decide between greedy (argmax) vs. top-p sampling:
        if GreedySearch {
            // --- Argmax branch: process each logits part in parallel ---
            let partialResults = try await withThrowingTaskGroup(of: PartialMax.self) { group -> [PartialMax] in
                for i in 1...splitLMHead {
                    let partIndex = i
                    let logitsKey = "logits\(partIndex)"
                    
                    guard let logitsPart = outputBackings[logitsKey] else {
                        throw InferenceError.inferenceError("Missing feature \(logitsKey)")
                    }
                    
                    group.addTask { @Sendable in
                        let localLogitsPart = logitsPart
                        let localOffset = (partIndex - 1) * logitsPart.count
                        
                        let buffer = try self.getFloatBuffer(from: localLogitsPart)
                        defer { buffer.unlock?() }
                        
                        #if arch(arm64)
                        let count = buffer.count
                        var localMaxValue: Float = -Float.infinity
                        var localMaxIndex = 0
                        
                        var start = 0
                        if localOffset == 0 && self.FilterLLAMA01 {
                            start = 2  // filtering special tokens
                            if self.debugLevel >= 2 {
                                print("Filtering special tokens: start=\(_padTokenId)")
                            }
                        }
                        
                        if let f16 = buffer.float16Ptr {
                            for j in start..<count {
                                let value = Float(f16[j])
                                if value > localMaxValue {
                                    localMaxValue = value
                                    localMaxIndex = localOffset + j
                                }
                            }
                        } else if let f32 = buffer.float32Ptr {
                            for j in start..<count {
                                let value = f32[j]
                                if value > localMaxValue {
                                    localMaxValue = value
                                    localMaxIndex = localOffset + j
                                }
                            }
                        } else {
                            throw InferenceError.inferenceError("No logits buffer for \(logitsKey)")
                        }
                        return PartialMax(value: localMaxValue, index: localMaxIndex)
                        #else
                        fatalError("Unsupported architecture, only Apple Silicon is supported")
                        #endif
                    }
                }
                
                var results: [PartialMax] = []
                for try await result in group {
                    results.append(result)
                }
                return results
            }
            
            let globalMax = partialResults.reduce(PartialMax(value: -Float.infinity, index: 0)) { current, next in
                next.value > current.value ? next : current
            }
            
            if debugLevel >= 1 {
                print("\nArgmax token:", globalMax.index)
                print("Argmax value:", globalMax.value)
            }
            return globalMax.index
        } else {
            // --- Optimized sparse sampling: work directly with (index, logit) pairs ---
            let logitsResults = try await withThrowingTaskGroup(of: [(Int, Float)].self) { group -> [[(Int, Float)]] in
                for i in 1...splitLMHead {
                    let partIndex = i
                    let logitsKey = "logits\(partIndex)"
                    guard let logitsPart = outputBackings[logitsKey] else {
                        throw InferenceError.inferenceError("Missing feature \(logitsKey)")
                    }
                    group.addTask { @Sendable in
                        let localLogitsPart = logitsPart
                        let localOffset = (partIndex - 1) * logitsPart.count
                        
                        let buffer = try self.getFloatBuffer(from: localLogitsPart)
                        defer { buffer.unlock?() }
                        
                        #if arch(arm64)
                        let count = buffer.count
                        
                        // Only keep top candidates from this chunk - avoid huge arrays
                        var topCandidates: [(Int, Float)] = []
                        let chunkK = 100  // Keep reasonable number per chunk
                        topCandidates.reserveCapacity(chunkK)
                        
                        var start = 0
                        if localOffset == 0 && self.FilterLLAMA01 {
                            start = 2
                        }
                        
                        if let f16 = buffer.float16Ptr {
                            for j in start..<count {
                                let value = Float(f16[j])
                                let globalIndex = localOffset + j
                                
                                if topCandidates.count < chunkK {
                                    topCandidates.append((globalIndex, value))
                                    if topCandidates.count == chunkK {
                                        topCandidates.sort { $0.1 > $1.1 }
                                    }
                                } else if value > topCandidates[chunkK - 1].1 {
                                    topCandidates[chunkK - 1] = (globalIndex, value)
                                    // Bubble up
                                    var idx = chunkK - 1
                                    while idx > 0 && topCandidates[idx].1 > topCandidates[idx - 1].1 {
                                        topCandidates.swapAt(idx, idx - 1)
                                        idx -= 1
                                    }
                                }
                            }
                        } else if let f32 = buffer.float32Ptr {
                            for j in start..<count {
                                let value = f32[j]
                                let globalIndex = localOffset + j
                                
                                if topCandidates.count < chunkK {
                                    topCandidates.append((globalIndex, value))
                                    if topCandidates.count == chunkK {
                                        topCandidates.sort { $0.1 > $1.1 }
                                    }
                                } else if value > topCandidates[chunkK - 1].1 {
                                    topCandidates[chunkK - 1] = (globalIndex, value)
                                    // Bubble up
                                    var idx = chunkK - 1
                                    while idx > 0 && topCandidates[idx].1 > topCandidates[idx - 1].1 {
                                        topCandidates.swapAt(idx, idx - 1)
                                        idx -= 1
                                    }
                                }
                            }
                        } else {
                            throw InferenceError.inferenceError("No logits buffer for \(logitsKey)")
                        }
                        return topCandidates
                        #else
                        fatalError("Unsupported architecture, only Apple Silicon is supported")
                        #endif
                    }
                }
                
                var allLogits: [[(Int, Float)]] = []
                for try await logits in group {
                    allLogits.append(logits)
                }
                return allLogits
            }
            
            // Flatten to sparse representation (index, logit) pairs
            var sparseLogits = logitsResults.flatMap { $0 }
            
            // Apply repetition penalty directly to sparse data
            if samplingConfig.repetitionPenalty != 1.0 && !generatedTokenHistory.isEmpty {
                let penaltyTokens = Set(generatedTokenHistory)
                for i in 0..<sparseLogits.count {
                    let (tokenId, logit) = sparseLogits[i]
                    if penaltyTokens.contains(tokenId) {
                        if logit < 0 {
                            sparseLogits[i].1 = logit * Float(samplingConfig.repetitionPenalty)
                        } else {
                            sparseLogits[i].1 = logit / Float(samplingConfig.repetitionPenalty)
                        }
                    }
                }
            }
            
            // Find max for numerical stability
            let maxLogit = sparseLogits.map { $0.1 }.max() ?? 0
            
            // Apply temperature and compute scores
            for i in 0..<sparseLogits.count {
                sparseLogits[i].1 = exp((sparseLogits[i].1 - maxLogit) / Float(samplingConfig.temperature))
            }
            
            // Apply top-k filtering if enabled
            if samplingConfig.topK > 0 && samplingConfig.topK < sparseLogits.count {
                // Use partial sort to get top-k efficiently
                sparseLogits.sort { $0.1 > $1.1 }
                sparseLogits = Array(sparseLogits.prefix(samplingConfig.topK))
            }
            
            // Apply top-p filtering if enabled
            if samplingConfig.topP < 1.0 {
                if samplingConfig.topK <= 0 {
                    // Sort if we haven't already done top-k
                    sparseLogits.sort { $0.1 > $1.1 }
                }
                
                let totalScore = sparseLogits.reduce(0) { $0 + $1.1 }
                let threshold = Float(samplingConfig.topP) * totalScore
                
                var cumulative: Float = 0.0
                var cutoffIndex = sparseLogits.count
                for (i, (_, score)) in sparseLogits.enumerated() {
                    cumulative += score
                    if cumulative >= threshold {
                        cutoffIndex = i + 1
                        break
                    }
                }
                sparseLogits = Array(sparseLogits.prefix(cutoffIndex))
            }
            
            // Sample from filtered candidates
            let totalScore = sparseLogits.reduce(0) { $0 + $1.1 }
            let r = Float.random(in: 0..<totalScore)
            
            var cumulative: Float = 0.0
            for (tokenId, score) in sparseLogits {
                cumulative += score
                if r <= cumulative {
                    if debugLevel >= 1 {
                        print("\nSampled token:", tokenId)
                    }
                    return tokenId
                }
            }
            
            // Fallback to highest scoring token
            return sparseLogits.first?.0 ?? 0
        }
     }

    /// Generates the next token using monolithic model - takes input_ids directly
    private func generateNextTokenMonolithic(
        for lastToken: Int,
        currentPos: Int,
        temperature: Float,
        tokenizer: Tokenizer? = nil
    ) async throws -> Int {
        // Safety check: ensure position is within bounds
        let safePos = currentPos - 1
        guard safePos >= 0 && safePos < contextLength else {
            throw InferenceError.inferenceError("Position \(safePos) out of bounds for context length \(contextLength)")
        }

        // Switch to infer_rotate after sliding-window boundary when available.
        let monolithicModel = try monolithicInferModel(for: safePos)

        // Create input tensor for single token
        let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenArray[[0, 0] as [NSNumber]] = NSNumber(value: lastToken)

        // Create position IDs
        let positionIds = try MLMultiArray(shape: [1], dataType: .int32)
        positionIds[0] = NSNumber(value: safePos)

        // Get causal mask for single token - use safePos (currentPos - 1) to match position_ids
        // At position N, we should see positions 0 to N (not 0 to N+1)
        let causalMask = try getCausalMask(for: 1, at: safePos)

        // Create current_pos tensor
        let currentPosArray = try MLMultiArray(shape: [1], dataType: .int32)
        currentPosArray[0] = NSNumber(value: safePos)

        // Create input feature provider - monolithic takes input_ids directly
        let inferInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": tokenArray,
            "position_ids": positionIds,
            "causal_mask": causalMask,
            "current_pos": currentPosArray
        ])

        // Use ring buffer (N=16) to avoid ANE race conditions.
        // Select buffer slot based on token counter modulo ring depth.
        let bufferSlot = monolithicTokenCounter % monolithicRingBufferDepth
        monolithicTokenCounter += 1

        // For both argmax and logits mode, use pre-allocated output backings
        let inferOptions = MLPredictionOptions()

        if argmaxInModel {
            // Argmax mode: serial queue + output backings for synchronization
            // Same queue as prefill ensures consistent execution context
            guard bufferSlot < monolithicOutputBackingsRing.count else {
                throw InferenceError.inferenceError("Ring buffer not initialized for argmax mode")
            }
            let currentBackings = monolithicOutputBackingsRing[bufferSlot]
            inferOptions.outputBackings = currentBackings

            // Run prediction on same serial queue as prefill
            var predictionError: Error?
            predictionQueue.sync { [self] in
                do {
                    _ = try monolithicModel.prediction(from: inferInput, using: state, options: inferOptions)
                } catch {
                    predictionError = error
                }
            }
            if let error = predictionError {
                throw error
            }
            maybeDelayBeforeReadingPredictionOutputsSync()

            // Read from backings - data is synced after prediction completes
            guard let idxArray = currentBackings["argmax_idx"],
                  let valArray = currentBackings["argmax_val"] else {
                throw InferenceError.inferenceError("Missing argmax_idx or argmax_val in backings")
            }

            // Find the chunk with highest value
            var bestChunk = 0
            var bestLocalIdx = 0
            var bestVal: Float = -Float.infinity
            let numChunks = idxArray.count
            let layout = resolveArgmaxChunkLayout(numChunks: numChunks, idxArray: idxArray)
            let chunkSizes = layout.sizes
            let chunkOffsets = layout.offsets

            // Collect all values for debug output
            var chunkData: [(chunk: Int, localIdx: Int, val: Float)] = []

            for i in 0..<numChunks {
                let localIdx = idxArray[i].intValue
                let chunkVal = valArray[i].floatValue
                chunkData.append((chunk: i, localIdx: localIdx, val: chunkVal))

                if chunkVal > bestVal {
                    bestVal = chunkVal
                    bestChunk = i
                    bestLocalIdx = localIdx
                }
            }

            // Compute global token ID: local_idx + (chunk * chunk_size)
            let globalIdx = bestLocalIdx + chunkOffsets[bestChunk]

            // Debug output (similar to Python's --debug-argmax)
            if debugLevel >= 1 {
                print("\n=== Argmax Debug (Swift) ===")
                print("argmax_idx count: \(numChunks), argmax_val count: \(numChunks)")
                print("Per-chunk results (LOCAL indices):")

                // Sort by value to find top-3
                let sortedByVal = chunkData.sorted { $0.val > $1.val }
                let top3Chunks = Set(sortedByVal.prefix(3).map { $0.chunk })

                var anyOutOfRange = false
                for i in 0..<numChunks {
                    let local = chunkData[i].localIdx
                    let val = chunkData[i].val
                    let computedGlobal = local + chunkOffsets[i]
                    let inRange = local >= 0 && local < chunkSizes[i]
                    if !inRange { anyOutOfRange = true }

                    var marker = ""
                    if i == bestChunk { marker += " <-- SELECTED" }
                    if top3Chunks.contains(i) && i != bestChunk {
                        if let rank = sortedByVal.firstIndex(where: { $0.chunk == i }) {
                            marker += " (top-\(rank + 1))"
                        }
                    }
                    let rangeOk = inRange ? "✓" : "✗ (expected 0-\(chunkSizes[i]-1))"
                    print("  Chunk \(String(format: "%2d", i)): local=\(String(format: "%5d", local)), global=\(String(format: "%6d", computedGlobal)), offset=\(String(format: "%6d", chunkOffsets[i])), size=\(String(format: "%5d", chunkSizes[i])), val=\(String(format: "%8.4f", val)), range=\(rangeOk)\(marker)")
                }

                print("Result: best_chunk=\(bestChunk), local_idx=\(bestLocalIdx), global_idx=\(globalIdx), best_val=\(String(format: "%.4f", bestVal))")

                // Value comparison
                if sortedByVal.count >= 2 {
                    let valDiff = abs(sortedByVal[0].val - sortedByVal[1].val)
                    print("Value comparison: top-1=\(String(format: "%.6f", sortedByVal[0].val)), top-2=\(String(format: "%.6f", sortedByVal[1].val)), diff=\(String(format: "%.6f", valDiff))")
                    if valDiff < 0.01 {
                        print("  WARNING: Values are very close - possible precision issue!")
                    }
                }

                if anyOutOfRange {
                    print("⚠️ WARNING: Some local indices are outside expected per-chunk ranges.")
                }
            }

            return globalIdx
        }

        // Logits mode: use output backings
        guard bufferSlot < monolithicOutputBackingsRing.count else {
            throw InferenceError.inferenceError("Ring buffer not initialized properly")
        }
        let currentBackings = monolithicOutputBackingsRing[bufferSlot]

        // Ring buffer depth ensures buffer isn't reused while ANE is still writing.
        // IMPORTANT: Do NOT lock the CVPixelBuffer before prediction!
        inferOptions.outputBackings = currentBackings

        // Run prediction synchronously on serial queue to prevent ANE race conditions
        var predictionError: Error?
        predictionQueue.sync { [self] in
            do {
                _ = try monolithicModel.prediction(from: inferInput, using: state, options: inferOptions)
            } catch {
                predictionError = error
            }
        }
        if let error = predictionError {
            throw error
        }
        maybeDelayBeforeReadingPredictionOutputsSync()

        // Build per-logit split sizes/offsets from actual model outputs.
        // Qwen/Gemma splits are not guaranteed to be 16384-wide.
        var chunkSizes: [Int] = []
        chunkSizes.reserveCapacity(splitLMHead)
        var chunkOffsets: [Int] = []
        chunkOffsets.reserveCapacity(splitLMHead)
        var runningOffset = 0
        for i in 1...splitLMHead {
            let logitsKey = "logits\(i)"
            guard let logitsPart = currentBackings[logitsKey] else {
                throw InferenceError.inferenceError("Missing \(logitsKey)")
            }
            chunkOffsets.append(runningOffset)
            let size = logitsPart.count
            chunkSizes.append(size)
            runningOffset += size
        }
        let totalVocabSize = runningOffset
        let chunkOffsetsConst = chunkOffsets

        // Process logits - try Metal GPU argmax first, fallback to CPU
        if GreedySearch {
            // Try Metal argmax only for the fixed-layout case it currently supports.
            // MetalArgmax assumes 16 chunks x 16384 each.
            let metalCompatible = (splitLMHead == 16) && chunkSizes.allSatisfy { $0 == 16384 }
            if metalCompatible, let metal = metalArgmax {
                do {
                    let tokenId = try metal.findArgmax(
                        backings: currentBackings,
                        splitCount: splitLMHead,
                        vocabSize: totalVocabSize,
                        filterFirst: FilterLLAMA01
                    )
                    if debugLevel >= 1 {
                        print("\nMetal argmax token:", tokenId)
                    }
                    return tokenId
                } catch {
                    if debugLevel >= 1 {
                        print("Metal argmax failed: \(error), falling back to CPU")
                    }
                }
            }

            // Fallback: parallel CPU argmax with Accelerate SIMD
            let parallelFactor = 2
            let totalTasks = splitLMHead * parallelFactor
            var partialResults = [(Float, Int)](repeating: (-Float.infinity, 0), count: totalTasks)

            DispatchQueue.concurrentPerform(iterations: totalTasks) { taskIdx in
                let chunkIdx = taskIdx / parallelFactor
                let subIdx = taskIdx % parallelFactor
                let logitsKey = "logits\(chunkIdx + 1)"
                let chunkOffset = chunkOffsetsConst[chunkIdx]

                guard let logitsPart = currentBackings[logitsKey],
                      let buffer = try? self.getFloatBuffer(from: logitsPart) else {
                    return
                }
                defer { buffer.unlock?() }

                let totalCount = buffer.count
                let subChunkSize = totalCount / parallelFactor

                var subStart = subIdx * subChunkSize
                var subEnd = (subIdx == parallelFactor - 1) ? totalCount : (subIdx + 1) * subChunkSize

                if chunkOffset == 0 && subIdx == 0 && FilterLLAMA01 {
                    subStart = 2
                }

                let effectiveCount = subEnd - subStart
                if effectiveCount <= 0 {
                    return
                }

                if let f16 = buffer.float16Ptr {
                    var floatBuffer = [Float](repeating: 0, count: effectiveCount)

                    f16.advanced(by: subStart).withMemoryRebound(to: UInt16.self, capacity: effectiveCount) { uint16Ptr in
                        var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: uint16Ptr),
                                               height: 1, width: vImagePixelCount(effectiveCount),
                                               rowBytes: effectiveCount * 2)
                        floatBuffer.withUnsafeMutableBufferPointer { floatPtr in
                            var dst = vImage_Buffer(data: floatPtr.baseAddress!,
                                                   height: 1, width: vImagePixelCount(effectiveCount),
                                                   rowBytes: effectiveCount * 4)
                            vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
                        }
                    }

                    var maxValue: Float = -Float.infinity
                    var maxIndex: vDSP_Length = 0
                    vDSP_maxvi(floatBuffer, 1, &maxValue, &maxIndex, vDSP_Length(effectiveCount))

                    partialResults[taskIdx] = (maxValue, chunkOffset + subStart + Int(maxIndex))
                } else if let f32 = buffer.float32Ptr {
                    var maxValue: Float = -Float.infinity
                    var maxIndex: vDSP_Length = 0
                    vDSP_maxvi(f32.advanced(by: subStart), 1, &maxValue, &maxIndex, vDSP_Length(effectiveCount))
                    partialResults[taskIdx] = (maxValue, chunkOffset + subStart + Int(maxIndex))
                }
            }

            // Find global max from partial results
            var globalMaxValue: Float = -Float.infinity
            var globalMaxIndex = 0
            for (maxVal, maxIdx) in partialResults {
                if maxVal > globalMaxValue {
                    globalMaxValue = maxVal
                    globalMaxIndex = maxIdx
                }
            }

            if debugLevel >= 1 {
                print("\nMonolithic parallel argmax token:", globalMaxIndex)
            }
            return globalMaxIndex
        } else {
            // Sampling branch
            var allLogits: [Float] = []
            allLogits.reserveCapacity(totalVocabSize)

            for i in 1...splitLMHead {
                let logitsKey = "logits\(i)"
                guard let logitsPart = currentBackings[logitsKey] else {
                    throw InferenceError.inferenceError("Missing \(logitsKey)")
                }

                let buffer = try getFloatBuffer(from: logitsPart)
                defer { buffer.unlock?() }

                if let f16 = buffer.float16Ptr {
                    for j in 0..<buffer.count {
                        allLogits.append(Float(f16[j]))
                    }
                } else if let f32 = buffer.float32Ptr {
                    for j in 0..<buffer.count {
                        allLogits.append(f32[j])
                    }
                } else {
                    throw InferenceError.inferenceError("No logits buffer for \(logitsKey)")
                }
            }

            // Apply top-p sampling
            let sampledToken = topPSample(logits: allLogits, temperature: temperature, topP: Float(samplingConfig.topP))
            if debugLevel >= 1 {
                print("\nMonolithic sampled token:", sampledToken)
            }
            return sampledToken
        }
    }

    /// Synchronous argmax inference - eliminates async overhead for maximum performance.
    /// This is called directly from the inference loop when argmaxInModel is true.
    /// Returns the token ID synchronously without any async suspension/resume overhead.
    public func generateNextTokenArgmaxSync(
        for lastToken: Int,
        currentPos: Int
    ) throws -> (token: Int, score: Float) {
        guard argmaxInModel else {
            throw InferenceError.inferenceError("generateNextTokenArgmaxSync called but argmaxInModel is false")
        }

        // Safety check: ensure position is within bounds
        let safePos = currentPos - 1
        guard safePos >= 0 && safePos < contextLength else {
            throw InferenceError.inferenceError("Position \(safePos) out of bounds for context length \(contextLength)")
        }

        // Switch to infer_rotate after sliding-window boundary when available.
        let monolithicModel = try monolithicInferModel(for: safePos)

        // Use pre-allocated input arrays
        guard let tokenArray = argmaxTokenArray,
              let positionIds = argmaxPositionIds,
              let currentPosArray = argmaxCurrentPosArray else {
            throw InferenceError.inferenceError("Pre-allocated argmax input arrays not initialized")
        }

        // Update int32 values using direct pointer access
        tokenArray.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(lastToken)
        positionIds.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(safePos)
        currentPosArray.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(safePos)

        // Use pre-allocated causal mask with efficient single-value update
        let maskBuffer = argmaxCausalMaskBuffer
        let maskArray = argmaxCausalMask
        guard maskBuffer != nil || maskArray != nil else {
            throw InferenceError.inferenceError("Pre-allocated argmax causal mask not initialized")
        }

        // Only update the mask if position changed (incremental update)
        // For position N, we need positions 0..N to be 0 (visible), rest -inf
        if lastArgmaxPosition != safePos {
            // Set the new position to 0 (make visible) using direct pixel buffer access
            if safePos < contextLength {
                if let maskBuffer {
                    CVPixelBufferLockBaseAddress(maskBuffer, [])
                    if let baseAddress = CVPixelBufferGetBaseAddress(maskBuffer) {
                        baseAddress.assumingMemoryBound(to: Float16.self)[safePos] = Float16(0.0)
                    }
                    CVPixelBufferUnlockBaseAddress(maskBuffer, [])
                } else if let maskArray {
                    maskArray.dataPointer.assumingMemoryBound(to: Float16.self)[safePos] = Float16(0.0)
                }
            }
            lastArgmaxPosition = safePos
        }

        // Use pre-allocated input feature provider (values already updated in backing arrays)
        guard let inferInput = argmaxInferInput else {
            throw InferenceError.inferenceError("Pre-allocated argmax input provider not initialized")
        }

        // Use ring buffer to avoid ANE race conditions
        let bufferSlot = monolithicTokenCounter % monolithicRingBufferDepth
        monolithicTokenCounter += 1

        // Get output backings from ring buffer
        guard bufferSlot < monolithicOutputBackingsRing.count else {
            throw InferenceError.inferenceError("Ring buffer not initialized for argmax mode")
        }
        let currentBackings = monolithicOutputBackingsRing[bufferSlot]

        // Set output backings
        guard let inferOptions = argmaxInferOptions else {
            throw InferenceError.inferenceError("Pre-allocated argmax options not initialized")
        }
        inferOptions.outputBackings = currentBackings

        // Run on the same serial lane as other stateful predictions.
        try runStatefulPredictionOnQueue(
            model: monolithicModel,
            input: inferInput,
            options: inferOptions
        )
        maybeDelayBeforeReadingPredictionOutputsSync()

        // Read from output backings (data is stable now)
        guard let idxArray = currentBackings["argmax_idx"],
              let valArray = currentBackings["argmax_val"] else {
            throw InferenceError.inferenceError("Missing argmax backings")
        }

        // Find the chunk with highest value using direct pointer access
        let numChunks = idxArray.count
        let layout = resolveArgmaxChunkLayout(numChunks: numChunks, idxArray: idxArray)
        let chunkOffsets = layout.offsets

        var bestChunk = 0
        var bestLocalIdx = 0
        var bestVal: Float = -Float.infinity

        // Use direct pointer access for performance
        let idxPtr = idxArray.dataPointer.assumingMemoryBound(to: Int32.self)
        let valPtr = valArray.dataPointer.assumingMemoryBound(to: Float16.self)

        for i in 0..<numChunks {
            let chunkVal = Float(valPtr[i])
            if chunkVal > bestVal {
                bestVal = chunkVal
                bestChunk = i
                bestLocalIdx = Int(idxPtr[i])
            }
        }

        // Compute global token ID: local_idx + (chunk * chunk_size)
        let globalIdx = bestLocalIdx + chunkOffsets[bestChunk]

        return (globalIdx, bestVal)
    }

    /// Shifts the context window if needed (similar to the Python code).
    public func shiftWindow(
        currentPos: Int,  
        contextTokens: inout [Int],
        onWindowShift: (() -> Void)? = nil
    ) throws {
        if currentPos >= contextLength - 2 {
            // Calculate shift to maintain full batches
            let maxBatches = contextLength / batchSize
            let desiredBatches = max(1, maxBatches - 2)  // Leave room for new tokens
            // Modified calculation to ensure we shift by no less than CONTEXT-PREFILL_BATCH
            // This prevents overflow on the last prefill operation
            let minSafeSize = max(1, contextLength - batchSize)
            let newSize = min(desiredBatches * batchSize, minSafeSize)
            
            if debugLevel >= 2 {
                print("\nShifting context window:")
                print("Current position: \(currentPos)")
                print("Context length: \(contextLength), Batch size: \(batchSize)")
                print("Min safe size: \(minSafeSize)")
                print("New size: \(newSize)")
            }
            
            // Shift window: keep only the last newSize tokens.
            let shiftedTokens = Array(contextTokens[(currentPos - newSize)..<currentPos])
            // Reset the context to all zeros, then write the shifted tokens at the beginning.
            contextTokens = Array(repeating: 0, count: contextLength)
            for i in 0..<shiftedTokens.count {
                contextTokens[i] = shiftedTokens[i]
            }
            
            // Call the window shift callback to notify listeners
            onWindowShift?()
        }
    }
    
    /// Main generation loop. Given an initial (padded) token sequence, run prefill once,
    /// then generate tokens one-by-one until maxTokens are produced or an EOS token is reached.
    ///
    public func isBusy() ->Bool {
        return busy;
    }
    
    public func generateResponse(
        initialTokens: [Int],
        temperature: Float,
        maxTokens: Int,
        eosTokens: [Int],  // Changed to array to support multiple EOS tokens
        tokenizer: Tokenizer,
        onToken: ((Int) -> Void)? = nil,
        onWindowShift: (() -> Void)? = nil
    ) async throws -> ([Int], TimeInterval, String) {
        
        var generatedTokens: [Int] = []
        let startTime = CFAbsoluteTimeGetCurrent()
        var stopReason = "max_tokens"

        if (busy) {
            print("Should not happen!!!!!")
            if let firstEos = eosTokens.first {
                generatedTokens.append(firstEos)
            }
            return  (generatedTokens, 0, "Inference is Busy")
        }

        let _padTokenId = tokenizer.padTokenId
        abort_generation = 0;
        busy = true
        
        // Clear token history for new generation (only when sampling)
        if !GreedySearch {
            generatedTokenHistory.removeAll()
        }

        // Reset KV cache state for new conversation turn
        // This ensures each generateResponse call starts fresh, which is required
        // when the full conversation is re-tokenized for each turn
        if let chunks = ffnChunks, !chunks.isEmpty {
            // For monolithic models, create state from inferModel (matching initState behavior)
            // This ensures state compatibility when switching between prefill and infer functions
            if isMonolithic {
                state = chunks[0].inferModel.makeState()
            } else {
                state = chunks[0].prefillModel.makeState()
            }
            lastArgmaxPosition = -1  // Reset argmax position tracking
        }

        do {

            if debugLevel >= 1 {
                print("\n=== EOS Token Setup ===")
                print("EOS token IDs: \(eosTokens)")
                for eos in eosTokens {
                    print("  \(eos): '\(tokenizer.decode(tokens: [eos], skipSpecialTokens: false))'")
                }
            }
            
            // Create mutable copy of initialTokens
            var contextTokens = initialTokens

            if contextTokens.isEmpty {
                busy = false
                return (generatedTokens, 0, "empty_input")
            }

            // Optional debug mode: disable batch prefill and build KV cache using infer-only path
            var currentPos: Int
            let prefillTime: TimeInterval
            if disablePrefill {
                if debugLevel >= 1 {
                    print("\n[Debug] Prefill disabled: using infer-only prefill over \(contextTokens.count) tokens")
                }
                var pos = 0
                while pos < contextTokens.count {
                    let _ = try await generateNextToken(
                        for: contextTokens[pos],
                        currentPos: pos + 1,
                        temperature: 0,
                        tokenizer: tokenizer
                    )
                    if debugLevel >= 1 {
                        print("  Debug infer prefill token at pos \(pos): \(contextTokens[pos])")
                    }
                    pos += 1
                }
                currentPos = contextTokens.count
                prefillTime = 0
            } else {
                // Run prefill with mutable copy
                currentPos = try await runPrefill(on: &contextTokens, contextPos: contextTokens.count, tokenizer: tokenizer)
                prefillTime = CFAbsoluteTimeGetCurrent() - startTime
            }
            
            var firstKVStateDivergencePosition: Int?
            var printedKVStateLegend = false

            while generatedTokens.count < maxTokens {
                // Check if we need to shift the context window
                if currentPos >= contextLength - 2 {
                    // Calculate shift to maintain full batches
                    let maxBatches = contextLength / batchSize
                    let desiredBatches = max(1, maxBatches - 2)  // Leave room for new tokens
                    // Modified calculation to ensure we shift by no less than CONTEXT-PREFILL_BATCH
                    // This prevents overflow on the last prefill operation
                    let minSafeSize = max(1, contextLength - batchSize)
                    let newSize = min(desiredBatches * batchSize, minSafeSize)
                    
                    if debugLevel >= 2 {
                        print("\nShifting context window:")
                        print("Current position: \(currentPos)")
                        print("Context length: \(contextLength), Batch size: \(batchSize)")
                        print("Min safe size: \(minSafeSize)")
                        print("New size: \(newSize)")
                    }
                    
                    // Keep only the last newSize tokens
                    let shiftedTokens = Array(contextTokens[(currentPos - newSize)..<currentPos])
                    contextTokens = Array(repeating: 0, count: contextLength)
                    for i in 0..<shiftedTokens.count {
                        contextTokens[i] = shiftedTokens[i]
                    }
                    
                    // Call the window shift callback to notify listeners
                    onWindowShift?()
                    
                    // Reset state and run prefill on shifted content
                    state = ffnChunks[0].prefillModel.makeState()
                    currentPos = try await runPrefill(on: &contextTokens, contextPos: newSize, tokenizer: tokenizer)
                    
                    if debugLevel >= 2 {
                        print("Window shifted. New position: \(currentPos)")
                    }
                }
                
                // Append new token to contextTokens if needed
                if currentPos >= contextTokens.count {
                    contextTokens.append(_padTokenId)  // Placeholder value
                }
                
                guard currentPos > 0 && currentPos < contextTokens.count else {
                    throw InferenceError.inferenceError("Invalid position \(currentPos) for context length \(contextTokens.count)")
                }
                
                if (abort_generation != 0 ) {
                    stopReason = "abort_generation"+String(abort_generation)
                    if debugLevel >= 1 {
                        print("\nStopping: abort_generation (\(abort_generation))")
                    }
                    break
                }

                if isMonolithic, let slidingWindow, currentPos >= slidingWindow, !monolithicHasRotationSupport {
                    stopReason = "sliding_window_requires_rotate"
                    if debugLevel >= 1 {
                        print("\nStopping: reached sliding_window=\(slidingWindow) without rotate functions (currentPos=\(currentPos))")
                    }
                    break
                }

                if debugRepeatInferCount >= 2 {
                    let repeatCount = min(debugRepeatInferCount, 4)
                    let prefixTokens = Array(contextTokens.prefix(currentPos))
                    let lastToken = prefixTokens[currentPos - 1]
                    let savedMonolithicTokenCounter = monolithicTokenCounter

                    if debugLevel >= 1 && !debugRepeatOnlyDivergence {
                        print("\n[DebugRepeat] Running \(repeatCount)x infer repeat at pos \(currentPos - 1)")
                    }

                    var repeatTokens: [Int] = []
                    var repeatDurationsNs: [UInt64] = []
                    var repeatScores: [Float?] = []
                    var foundMatch = false
                    
                    // Store original debug level
                    let savedDebugLevel = debugLevel
                    
                    // Storage for hidden state comparisons
                    var capturedEmbeddings: [MLMultiArray?] = []
                    var capturedFinalHidden: [MLMultiArray?] = []
                    
                    // Capture KV cache state BEFORE first run
                    let initialKVSnapshot = captureKVCacheSnapshot()
                    
                    // We'll also save snapshots after each run to compare
                    var kvSnapshots: [[String: MLMultiArray]] = []

                    var kvInitialToRun1Similarity: Float?
                    var kvRun1ToRun2Similarity: Float?
                    var kvPerBufferRun1ToRun2: [String: Float] = [:]
                    var kvDivergentAfterRun1: [String] = []
                    var kvDivergentRun1ToRun2: [String] = []
                    var kvRun1ToRun2Diverged = false

                    for i in 0..<repeatCount {
                        // Enable detailed logging for divergence analysis on repeat runs
                        if i > 0 && !foundMatch {
                            debugLevel = max(savedDebugLevel, 1)  // Ensure at least level 1
                        }
                        
                        let startNs = DispatchTime.now().uptimeNanoseconds
                        let token: Int
                        
                        // Enable hidden state capture for non-monolithic models
                        let shouldCaptureHidden = !isMonolithic && argmaxInModel
                        
                        if argmaxInModel && isMonolithic {
                            let result = try generateNextTokenArgmaxSync(for: lastToken, currentPos: currentPos)
                            token = result.token
                            repeatScores.append(result.score)
                            capturedEmbeddings.append(nil)  // Not captured for monolithic
                            capturedFinalHidden.append(nil)
                            if debugLevel >= 1 && !debugRepeatOnlyDivergence {
                                print("[DebugRepeat] argmax score=\(result.score)")
                            }
                        } else {
                            token = try await generateNextToken(
                                for: lastToken,
                                currentPos: currentPos,
                                temperature: temperature,
                                tokenizer: tokenizer,
                                captureHiddenStates: shouldCaptureHidden
                            )
                            repeatScores.append(nil)
                            
                            // Capture hidden states if available
                            if shouldCaptureHidden {
                                capturedEmbeddings.append(debugCapturedEmbeddings)
                                capturedFinalHidden.append(debugCapturedFinalHidden)
                            } else {
                                capturedEmbeddings.append(nil)
                                capturedFinalHidden.append(nil)
                            }
                        }
                        let endNs = DispatchTime.now().uptimeNanoseconds
                        let durationNs = endNs - startNs
                        let durationUs = durationNs / 1_000

                        repeatTokens.append(token)
                        repeatDurationsNs.append(durationNs)
                        
                        // Capture KV cache state AFTER this run
                        kvSnapshots.append(captureKVCacheSnapshot())

                        if debugLevel >= 1 && !debugRepeatOnlyDivergence {
                            let decoded = tokenizer.decode(tokens: [token], skipSpecialTokens: false)
                            print("[DebugRepeat] run \(i + 1): token=\(token) (\(decoded)) durationUs=\(durationUs)")
                        }

                        if i == 1, repeatTokens[0] == repeatTokens[1] {
                            if debugLevel >= 1 && !debugRepeatOnlyDivergence {
                                print("[DebugRepeat] match on run2 (t1 == t2)")
                            }
                            foundMatch = true
                            break
                        }
                        if i == 2, token == repeatTokens[0] || token == repeatTokens[1] {
                            if debugLevel >= 1 && !debugRepeatOnlyDivergence {
                                print("[DebugRepeat] match on run3 (t3 == t1 or t2)")
                            }
                            foundMatch = true
                            break
                        }
                        if i == 3 {
                            let matches = repeatTokens.dropLast().contains(token)
                            if debugLevel >= 1 && !debugRepeatOnlyDivergence {
                                if matches {
                                    print("[DebugRepeat] match on run4 (t4 matches a previous token)")
                                } else {
                                    print("[DebugRepeat] no match after run4")
                                }
                            }
                            foundMatch = matches
                        }
                    }
                    
                    // Restore debug level
                    debugLevel = savedDebugLevel

                    let tokenDiverged = !foundMatch && repeatTokens.count >= repeatCount

                    if (debugCompareKVStateEveryToken || tokenDiverged), kvSnapshots.count >= 2 {
                        let (sim0, _, divergent0) = compareKVCacheSnapshots(initialKVSnapshot, kvSnapshots[0])
                        let (sim12, perBuffer12, divergent12) = compareKVCacheSnapshots(kvSnapshots[0], kvSnapshots[1])

                        kvInitialToRun1Similarity = sim0
                        kvRun1ToRun2Similarity = sim12
                        kvPerBufferRun1ToRun2 = perBuffer12
                        kvDivergentAfterRun1 = divergent0
                        kvDivergentRun1ToRun2 = divergent12
                        kvRun1ToRun2Diverged = sim12 < kvStateSimilarityThreshold || !divergent12.isEmpty

                        if debugCompareKVStateEveryToken {
                            if !printedKVStateLegend {
                                print("[DebugRepeat][State] note: initial->run1 reflects expected KV write; divergence is judged by run1->run2")
                                printedKVStateLegend = true
                            }
                            // Always print state similarity for every repeated token so small drift is visible early.
                            if initialKVSnapshot.isEmpty || kvSnapshots[0].isEmpty || kvSnapshots[1].isEmpty {
                                print("[DebugRepeat][State] pos \(currentPos - 1) no KV state buffers captured")
                            } else {
                                print("[DebugRepeat][State] pos \(currentPos - 1) initial->run1=\(String(format: "%.8f", sim0)) run1->run2=\(String(format: "%.8f", sim12))")

                                if let worst = perBuffer12.min(by: { $0.value < $1.value }) {
                                    print("[DebugRepeat][State] worst run1->run2 buffer: \(worst.key)=\(String(format: "%.8f", worst.value))")
                                }
                            }

                            if kvRun1ToRun2Diverged {
                                if firstKVStateDivergencePosition == nil {
                                    firstKVStateDivergencePosition = currentPos - 1
                                    print("\n[DebugRepeat][State] first KV divergence observed at pos \(currentPos - 1)")
                                }

                                if !tokenDiverged {
                                    print("\n" + String(repeating: "-", count: 80))
                                    print("🟠 KV STATE DIVERGENCE (tokens still matched) at position \(currentPos - 1)")
                                    print(String(repeating: "-", count: 80))
                                }

                                if !kvDivergentRun1ToRun2.isEmpty {
                                    print("[DebugRepeat][State] divergent buffers: \(kvDivergentRun1ToRun2.prefix(6).joined(separator: ", "))\(kvDivergentRun1ToRun2.count > 6 ? "... (\(kvDivergentRun1ToRun2.count) total)" : "")")
                                }

                                if !tokenDiverged {
                                    printKVBlockDivergenceAnalysis(
                                        previousSnapshot: kvSnapshots[0],
                                        currentSnapshot: kvSnapshots[1],
                                        divergentBuffers: kvDivergentRun1ToRun2,
                                        focusTokenIndex: currentPos - 1
                                    )
                                }
                            }
                        }
                    }

                    if tokenDiverged {
                        if debugLevel >= 1 {
                            let decoded = tokenizer.decode(tokens: repeatTokens, skipSpecialTokens: false)
                            let runsDetails = repeatTokens.enumerated().map { idx, tok -> String in
                                let dec = tokenizer.decode(tokens: [tok], skipSpecialTokens: false)
                                let durUs = idx < repeatDurationsNs.count ? repeatDurationsNs[idx] / 1_000 : 0
                                let score = (argmaxInModel && isMonolithic && idx < repeatScores.count) ? repeatScores[idx] : nil
                                if let s = score {
                                    return "#\(idx + 1) tok=\(tok) \"\(dec)\" durUs=\(durUs) score=\(String(format: "%.4f", s))"
                                } else {
                                    return "#\(idx + 1) tok=\(tok) \"\(dec)\" durUs=\(durUs)"
                                }
                            }.joined(separator: "; ")

                            print("\n" + String(repeating: "=", count: 80))
                            print("🔴 DIVERGENCE DETECTED at position \(currentPos - 1)")
                            print(String(repeating: "=", count: 80))
                            print("Tokens: \(repeatTokens)")
                            print("Decoded: \(decoded)")
                            print("Runs: [\(runsDetails)]")
                            
                            // Compare hidden states if captured
                            if capturedEmbeddings.count >= 2, 
                               let embed0 = capturedEmbeddings[0],
                               let embed1 = capturedEmbeddings[1] {
                                let embedSim = cosineSimilarity(embed0, embed1)
                                print("\n📊 Hidden State Analysis:")
                                print("  Embedding cosine similarity (run1 vs run2): \(String(format: "%.8f", embedSim))")
                                
                                if embedSim < hiddenStateSimilarityThreshold {
                                    print("  ⚠️ Embeddings diverged! (< \(hiddenStateSimilarityThreshold))")
                                } else {
                                    print("  ✅ Embeddings identical")
                                }
                            }
                            
                            if capturedFinalHidden.count >= 2,
                               let final0 = capturedFinalHidden[0],
                               let final1 = capturedFinalHidden[1] {
                               let finalSim = cosineSimilarity(final0, final1)
                                print("  Final hidden cosine similarity (run1 vs run2): \(String(format: "%.8f", finalSim))")
                                
                                if finalSim < hiddenStateSimilarityThreshold {
                                    print("  ⚠️ Final hidden states diverged! (< \(hiddenStateSimilarityThreshold))")
                                    print("  → Issue is in FFN/state processing")
                                } else {
                                    print("  ✅ Final hidden states identical")
                                    print("  → Issue is in LM head argmax computation")
                                }
                            }
                            
                            // Additional KV cache analysis - compare actual state buffers
                            // Check if KV cache was modified DURING the inference runs
                            if let sim0 = kvInitialToRun1Similarity,
                               let sim12 = kvRun1ToRun2Similarity {
                                print("\n🔍 KV Cache State Buffer Analysis:")
                                print("  Initial → After Run1: \(String(format: "%.8f", sim0))")

                                if !kvDivergentAfterRun1.isEmpty && debugLevel >= 2 {
                                    print("    Divergent buffers after run1: \(kvDivergentAfterRun1.prefix(3).joined(separator: ", "))\(kvDivergentAfterRun1.count > 3 ? "... (\(kvDivergentAfterRun1.count) total)" : "")")
                                }

                                print("  After Run1 → After Run2: \(String(format: "%.8f", sim12))")
                                
                                if !kvDivergentRun1ToRun2.isEmpty && debugLevel >= 2 {
                                    print("    Divergent buffers: \(kvDivergentRun1ToRun2.prefix(3).joined(separator: ", "))\(kvDivergentRun1ToRun2.count > 3 ? "... (\(kvDivergentRun1ToRun2.count) total)" : "")")
                                }
                                
                                // Analysis
                                if kvRun1ToRun2Diverged {
                                    print("  🔴 CRITICAL: KV cache state diverged between identical runs!")
                                    print("  → State buffers are being modified non-deterministically")
                                    print("  → This is an iOS ANE bug: same input + same initial state ≠ same final state")
                                    
                                    if debugLevel >= 2 && !kvDivergentRun1ToRun2.isEmpty {
                                        print("\n  Per-buffer divergence analysis:")
                                        for bufferName in kvDivergentRun1ToRun2.prefix(5) {
                                            if let similarity = kvPerBufferRun1ToRun2[bufferName] {
                                                print("    \(bufferName): \(String(format: "%.8f", similarity))")
                                            }
                                        }
                                    }

                                    printKVBlockDivergenceAnalysis(
                                        previousSnapshot: kvSnapshots[0],
                                        currentSnapshot: kvSnapshots[1],
                                        divergentBuffers: kvDivergentRun1ToRun2,
                                        focusTokenIndex: currentPos - 1
                                    )
                                } else {
                                    print("  ✅ KV cache state is deterministic")
                                    print("  → State buffers are identical after both runs")
                                    print("  → Divergence must be in the forward pass computation (not state)")
                                }
                            }
                            
                            print(String(repeating: "=", count: 80) + "\n")
                        }
                    }
                    monolithicTokenCounter = savedMonolithicTokenCounter
                }

                // Use synchronous path for argmax mode (eliminates async overhead for ~10% speedup)
                // Async path is used for logits mode which needs sampling
                let nextToken: Int
                if argmaxInModel && isMonolithic {
                    let result = try generateNextTokenArgmaxSync(
                        for: contextTokens[currentPos - 1],
                        currentPos: currentPos
                    )
                    nextToken = result.token
                    if debugLevel >= 1 && !debugRepeatOnlyDivergence {
                        print("[Argmax] token=\(nextToken) score=\(result.score)")
                    }
                } else {
                    nextToken = try await generateNextToken(
                        for: contextTokens[currentPos - 1],
                        currentPos: currentPos,
                        temperature: temperature,
                        tokenizer: tokenizer
                    )
                }
                
                // Debug token comparison
                if debugLevel >= 1 {
                    print("\nToken check:")
                    print("Next token: \(nextToken)")
                    print("Decoded: '\(tokenizer.decode(tokens: [nextToken], skipSpecialTokens: false))'")
                    print("Is EOS? \(eosTokens.contains(nextToken))")
                }

                // Check for stop tokens before adding to generated tokens
                if eosTokens.contains(nextToken) {
                    stopReason = "eos_token"
                    if debugLevel >= 1 {
                        print("\nStopping: EOS token detected (\(nextToken))")
                    }
                    break
                }
                
                // Only add token and continue if not a stop token
                generatedTokens.append(nextToken)
                if !GreedySearch {
                    generatedTokenHistory.append(nextToken)  // Track for repetition penalty only when sampling
                }
                contextTokens[currentPos] = nextToken
                onToken?(nextToken)
                currentPos += 1
            }
            busy = false;
            return (generatedTokens, prefillTime, stopReason)
        } catch {
            print("\nError during generation: \(error)")
            busy = false;
            throw error
        }
    }
    
    private func debugHiddenStates(_ hidden_states: MLMultiArray, prefix: String) {
        if debugLevel >= 1 {
            print("\(prefix) shape: \(hidden_states.shape.map { $0.intValue })")
        }
        if debugLevel >= 2 {
            print("\(prefix) first 10 values: ", terminator: "")
            for i in 0..<min(10, hidden_states.count) {
                print(String(format: "%.4f", Float(truncating: hidden_states[i])), terminator: " ")
            }
            print()  // New line
        }
    }

    private func debugTensor(_ tensor: MLMultiArray, prefix: String, level: Int = 1) {
        if debugLevel >= level {
            print("\n\(prefix) shape:", tensor.shape.map { $0.intValue })
            
            if debugLevel >= 2 {
                print("First 10 values: ", terminator: "")
                for i in 0..<min(10, tensor.count) {
                    print(String(format: "%.4f", Float(truncating: tensor[i])), terminator: " ")
                }
                print("\nLast 10 values: ", terminator: "")
                for i in max(0, tensor.count-10)..<tensor.count {
                    print(String(format: "%.4f", Float(truncating: tensor[i])), terminator: " ")
                }
                print()
            }
        }
    }

    public func unload()
    {
        // Just clear our local reference - no need to set model's outputBackings
        lmheadOutputBackings = nil
        hiddenStatesBackings_emb = nil
        hiddenStatesBackings_ffn = nil
        hiddenStatesBackings_ffnPingPong.removeAll(keepingCapacity: false)
        hiddenStatesBackings_last = nil
        hiddenStatesBackings_lastPingPong.removeAll(keepingCapacity: false)
        hiddenStatesBackings_emb_prefill = nil
        hiddenStatesBackings_ffn_prefill = nil
        hiddenStatesBackings_ffn_prefillPingPong.removeAll(keepingCapacity: false)
        prefillUpdateMask = nil
        prefillUpdateMaskBuffer = nil
        state = nil
        embedModel = nil
        lmheadModel = nil
        ffnChunks = nil

    }
    deinit {
        unload()
    }
}

/// Custom errors for inference.
public enum InferenceError: Error {
    case missingLogits
    case inferenceError(String)
    case windowShiftError(String)
}
