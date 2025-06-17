import CoreVideo
@preconcurrency import CoreML
import CoreFoundation

/// Manages inference by wrapping a CoreML model and handling state.
@preconcurrency public final class InferenceManager: @unchecked Sendable {
    private var hidden_states: Int = -1
    private var embedModel: MLModel!
    private var lmheadModel: MLModel!
    private var ffnChunks: [FFNChunk]!  // Use the FFNChunk defined in FFNChunk.swift
    private var state: MLState!
    private let contextLength: Int
    private let batchSize: Int
    private let fullCausalMask: MLMultiArray  // We already have this
    private var debugLevel: Int
    private var v110: Bool = false // old conversio has batch x hidden_states for the last chunk
    // Change timing property to CFAbsoluteTime
    private var prefillEndTime: CFAbsoluteTime?
    private var FilterLLAMA01: Bool = false
    private let splitLMHead: Int
    
    private var lmheadOutputBackings: [String: MLMultiArray]?
    private var hiddenStatesBackings_emb: [String: MLMultiArray]?  // For embed output
    private var hiddenStatesBackings_ffn: [String: MLMultiArray]?  // For FFN input/output
    private var hiddenStatesBackings_last: [String: MLMultiArray]?  // Prefill the last chunk
    private var hiddenStatesBackings_emb_prefill: [String: MLMultiArray]?  // For embed output in prefill
    private var hiddenStatesBackings_ffn_prefill: [String: MLMultiArray]?  // For FFN output in prefill
    private var GreedySearch = true
    nonisolated(unsafe) private var abort_generation = Int(0)
    private var _busy = false
    private var busy: Bool {
        get { _busy }
        set { _busy = newValue }
    }
    
    // Move struct definition to class scope, before the methods
    private struct PartialMax {
        let value: Float
        let index: Int
    }
    
    public func AbortGeneration( Code : Int)
    {
        abort_generation = Code
    }
    
    public func set_FilterLLAMA01(value: Bool)
    {
        FilterLLAMA01 = value
    }


    public init(models: LoadedModels, contextLength: Int, batchSize: Int, splitLMHead: Int = 8, debugLevel: Int = 0, v110: Bool = false) throws {  // Make init throwing
        self.debugLevel = debugLevel
        self.embedModel = models.embedModel
        self.lmheadModel = models.lmheadModel
        // Assume models.ffnChunks is available (see note below)
        self.ffnChunks = models.ffnChunks
        self.contextLength = contextLength
        self.batchSize = batchSize
        self.splitLMHead = splitLMHead
        self.v110 = v110 // Set the v110 flag based on the parameter

        
        print("InferenceManager initialized with v110=\(v110), splitLMHead=\(splitLMHead), batchSize=\(batchSize)")
        self.fullCausalMask = try MLMultiArray(shape: [1, 1, NSNumber(value: contextLength), NSNumber(value: contextLength)], dataType: .float16)

        self.initState()

        initFullCausalMask()
        
        try initializeBackings()
        
        // Debug model descriptions
        if debugLevel >= 1 {
            print("\nLM Head Model Output Description:")
            for (name, desc) in lmheadModel.modelDescription.outputDescriptionsByName {
                print("Output \(name):")
                print("- Type: \(type(of: desc.type))")
                print("- Description: \(desc.type)")
            }
        }
        
    }
    
    public func initializeBackings() throws {
        // Initialize output backings for lmhead
        try initializeLMHeadOutputBackings()
        
        // Initialize hidden states backings
        try initializeHiddenStatesBackings()

        try initializePrefillBackings()
        try initializeLastChunkBacking()
    }
    
    
    public func initFullCausalMask()  {
        // Create full causal mask once with -inf and 0.0 (we already have this)
        
        // Initialize all values to -inf first
        for i in 0..<fullCausalMask.count {
            fullCausalMask[i] = NSNumber(value: Float(-Float.infinity))
        }
        
        // Set values to 0.0 where j <= i (causal masking)
        for i in 0..<contextLength {
            for j in 0..<contextLength {
                if j <= i {
                    fullCausalMask[[0, 0, i, j] as [NSNumber]] = NSNumber(value: Float(0.0))
                }
            }
        }
    }
    
    public func initState()  {
        self.state = ffnChunks[0].prefillModel.makeState()
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
        let featureNames = (1...splitLMHead).map { i in "logits\(i)" }
        var outputBackingsDict: [String: MLMultiArray] = [:]
        
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
            
            // Calculate dimensions for pixel buffer
            let lastDim = shape.last?.intValue ?? 1
            let otherDims = shape.dropLast().reduce(1) { $0 * $1.intValue }
            
            // Create IOSurface-backed pixel buffer
            let attributes: [String: Any] = [
                //kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                kCVPixelBufferMetalCompatibilityKey as String: true
            ]
            
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
            
            let attributes: [String: Any] = [
                kCVPixelBufferMetalCompatibilityKey as String: true
            ]
            
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
            
            // Create FFN output backing
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
                throw InferenceError.inferenceError("Failed to create pixel buffer for FFN output")
            }
            
            // Store FFN input/output backing
            hiddenStatesBackings_ffn = ["output_hidden_states": MLMultiArray(pixelBuffer: ffnBuffer, shape: shape)]
        }
    }


    private func initializeLastChunkBacking() throws {
        guard let desc = ffnChunks.last?.prefillModel.modelDescription.outputDescriptionsByName["output_hidden_states"],
            let constraint = desc.multiArrayConstraint else {
            throw InferenceError.inferenceError("Failed to get last chunk output description")
        }
        
        let hiddenSize = constraint.shape.last?.intValue ?? self.hidden_states
    
        let shape: [NSNumber] = [1, 1, NSNumber(value: hiddenSize)]
        
        let attributes: [String: Any] = [
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            hiddenSize,  // width
            1,          // height (batch=1)
            kCVPixelFormatType_OneComponent16Half,
            attributes as CFDictionary,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw InferenceError.inferenceError("Failed to create last chunk pixel buffer: \(status)")
        }
        
        let backing = MLMultiArray(pixelBuffer: buffer, shape: shape)
        hiddenStatesBackings_last = ["output_hidden_states": backing]
        
        if debugLevel >= 1 {
            print("\nLast Chunk Backing Initialized:")
            print("Shape: \(shape.map { $0.intValue })")
        }
    }
    
    private func initializePrefillBackings() throws {
        let hiddenSize = self.hidden_states  // Adjust based on your model's hidden size
        let shape: [NSNumber] = [1, NSNumber(value: batchSize), NSNumber(value: hiddenSize)]
        let attributes: [String: Any] = [kCVPixelBufferMetalCompatibilityKey as String: true]
        
        if debugLevel >= 1 {
            print("\n=== Prefill Backing Initialization ===")
            print("Hidden size:", hiddenSize)
            print("Batch size:", batchSize)
            print("Prefill backing shape:", shape.map { $0.intValue })
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
        
        // FFN prefill backing
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
            throw InferenceError.inferenceError("Failed to create FFN prefill pixel buffer")
        }
        hiddenStatesBackings_ffn_prefill = ["output_hidden_states": MLMultiArray(pixelBuffer: ffnBuffer, shape: shape)]
    }
    
    // Helper to get causal mask slice for current position
    private func getCausalMask(for length: Int, at position: Int, paddingLength: Int? = nil) throws -> MLMultiArray {
        // Ensure position is within bounds
        let safePosition = min(position, contextLength - 1)
        
        // Create mask with correct dimensions
        let mask = try MLMultiArray(
            shape: [1, 1, NSNumber(value: length), NSNumber(value: contextLength)],
            dataType: .float16
        )
        
        // Fill mask with -inf by default
        for i in 0..<mask.count {
            mask[i] = NSNumber(value: Float(-Float.infinity))
        }
        
        // Set causal attention pattern
        for i in 0..<length {
            for j in 0..<contextLength {
                if j <= (safePosition + i) {
                    mask[[0, 0, i, j] as [NSNumber]] = NSNumber(value: Float(0.0))
                }
            }
        }
        
        // Apply padding if specified
        if let paddingLength = paddingLength {
            for i in paddingLength..<length {
                for j in 0..<contextLength {
                    mask[[0, 0, i, j] as [NSNumber]] = NSNumber(value: Float(-Float.infinity))
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
            debugTokens(Array(contextTokens.prefix(contextPos)), prefix: "Input")
        }
        guard let ffnChunks = ffnChunks else {
            throw InferenceError.inferenceError("ffnChunks was nil in runPrefill()")
        }
        var batchPos = 0
        while batchPos < contextPos {
            let batchEnd = min(batchPos + batchSize, contextPos)
            let currentBatchSize = batchEnd - batchPos
            
            if debugLevel >= 1 {
                print("\nPrefill batch: \(batchPos) to \(batchEnd), currentBatchSize: \(currentBatchSize)")
            }
            
            // Create input tensor for current batch
            let batchInput = try MLMultiArray(shape: [1, NSNumber(value: batchSize)], dataType: .int32)
            for i in 0..<currentBatchSize {
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
            
            for (index, chunk) in ffnChunks.enumerated() {
                let isLastChunk = index == chunkCount - 1
                let ffnOptions = MLPredictionOptions()
                
                if debugLevel >= 1 {
                    print("\nFFN chunk \(index + 1)/\(chunkCount), isLastChunk: \(isLastChunk)")
                    print("Current hidden states shape:", currentHiddenStates.shape.map { $0.intValue })
                }
                
                // Assign output backing BEFORE predict
                // During prefill, ALWAYS use prefill backing, never the last chunk backing
                if let backings = hiddenStatesBackings_ffn_prefill {
                    ffnOptions.outputBackings = backings  // Shape: [1, batch_size, hidden_states]
                    if debugLevel >= 1 {
                        print("Using FFN prefill backing with shape:", backings["output_hidden_states"]?.shape.map { $0.intValue } ?? [])
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
                
                // Run prediction with the assigned output backing
                _ = try await chunk.prefillModel.prediction(
                    from: prefillInput,
                    using: state,
                    options: ffnOptions
                )
                
                // Update currentHiddenStates - during prefill always use prefill backing
                guard let nextHiddenStates = hiddenStatesBackings_ffn_prefill?["output_hidden_states"] else {
                    throw InferenceError.inferenceError("Missing FFN prefill output backing")
                }
                currentHiddenStates = nextHiddenStates  // Shape: [1, batch_size, hidden_states]
                
                if debugLevel >= 2 {
                    debugTensor(currentHiddenStates, prefix: "FFN chunk \(index + 1) output")
                }
            }
            
            batchPos = batchEnd
        }
        
        return contextPos
    }

    func topPSample(logits: [Float], temperature: Float = 1.0, topP: Float = 0.9) -> Int {
        // Apply temperature scaling
        let scaledLogits = logits.map { $0 / temperature }
        
        // Compute softmax probabilities (with numerical stability)
        let maxLogit = scaledLogits.max() ?? 0
        let expLogits = scaledLogits.map { exp($0 - maxLogit) }
        let sumExp = expLogits.reduce(0, +)
        let probs = expLogits.map { $0 / sumExp }
        
        // Sort indices by descending probability
        let sorted = probs.enumerated().sorted { $0.element > $1.element }
        var cumulative: Float = 0.0
        var filtered: [(Int, Float)] = []
        for (index, prob) in sorted {
            cumulative += prob
            filtered.append((index, prob))
            if cumulative >= topP {
                break
            }
        }
        
        // Normalize the filtered probability subset
        let filteredSum = filtered.map { $0.1 }.reduce(0, +)
        let normalized = filtered.map { ($0.0, $0.1 / filteredSum) }
        
        // Sample from the normalized set
        let r = Float.random(in: 0..<1)
        var cumulativeSample: Float = 0.0
        for (index, prob) in normalized {
            cumulativeSample += prob
            if r <= cumulativeSample {
                return index
            }
        }
        
        return normalized.last!.0  // fallback (should not usually happen)
    }

    /// Generates the next token given the current token. This method calls the embedding model,
    /// passes the output through each FFN chunk's infer function, and then runs the LM head.
    public func generateNextToken(
        for lastToken: Int,
        currentPos: Int,
        temperature: Float,
        tokenizer: Tokenizer? = nil
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

        // Get hidden states from embed backing
        guard let hiddenStates = hiddenStatesBackings_emb?["hidden_states"] else {
            throw InferenceError.inferenceError("Missing embed output backing")
        }

        // Create position IDs (1D)
        let positionIds = try MLMultiArray(shape: [1], dataType: .int32)
        positionIds[0] = NSNumber(value: currentPos-1)

        // Get causal mask for single token at the correct position
        let causalMask = try getCausalMask(for: 1, at: currentPos)

        // Create current_pos as tensor
        let currentPosArray = try MLMultiArray(shape: [1], dataType: .int32)
        currentPosArray[0] = NSNumber(value: currentPos-1)

        // Run through FFN chunks using FFN backing
        var currentHiddenStates = hiddenStates

        for chunk in ffnChunks {
            let ffnOptions = MLPredictionOptions()
            if let backings = hiddenStatesBackings_ffn {
                ffnOptions.outputBackings = backings
            }
            
            let inferInput = try MLDictionaryFeatureProvider(dictionary: [
                "hidden_states": currentHiddenStates,
                "position_ids": positionIds,
                "causal_mask": causalMask,
                "current_pos": currentPosArray
            ])
            
            let _ = try await chunk.inferModel.prediction(from: inferInput, using: state, options: ffnOptions)
            
            guard let nextHiddenStates = hiddenStatesBackings_ffn?["output_hidden_states"] else {
                throw InferenceError.inferenceError("Missing FFN output backing")
            }
            currentHiddenStates = nextHiddenStates
        }

        debugHiddenStates(currentHiddenStates, prefix: "Final hidden states to LM head")
        
        // Run LM head with final hidden states
        let lmOptions = MLPredictionOptions()
        if let backings = lmheadOutputBackings {
            lmOptions.outputBackings = backings
        }

        let lmInput = try MLDictionaryFeatureProvider(dictionary: ["hidden_states": currentHiddenStates])
        let _ = try await lmheadModel.prediction(from: lmInput, options: lmOptions)
        
        guard let outputBackings = lmheadOutputBackings else {
            throw InferenceError.inferenceError("Output backings not initialized")
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
                        
                        guard let pixelBuffer = localLogitsPart.pixelBuffer else {
                            throw InferenceError.inferenceError("Missing or invalid \(logitsKey) in output backings")
                        }
                        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
                        defer {
                            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
                        }
                        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                            throw InferenceError.inferenceError("Could not get base address for \(logitsKey)")
                        }
                        
                        #if arch(arm64)
                        let buffer = baseAddress.assumingMemoryBound(to: Float16.self)
                        let count = localLogitsPart.count
                        var localMaxValue: Float = -Float.infinity
                        var localMaxIndex = 0
                        
                        var start = 0
                        if localOffset == 0 && self.FilterLLAMA01 {
                            start = 2  // filtering special tokens
                            if self.debugLevel >= 2 {
                                print("Filtering special tokens: start=\(_padTokenId)")
                            }
                        }
                        
                        for j in start..<count {
                            let value = Float(buffer[j])
                            if value > localMaxValue {
                                localMaxValue = value
                                localMaxIndex = localOffset + j
                            }
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
            // --- Top-p sampling branch: collect logits parts in parallel ---
            let logitsResults = try await withThrowingTaskGroup(of: [(Int, Float)].self) { group -> [[(Int, Float)]] in
                for i in 1...8 {
                    let partIndex = i
                    let logitsKey = "logits\(partIndex)"
                    guard let logitsPart = outputBackings[logitsKey] else {
                        throw InferenceError.inferenceError("Missing feature \(logitsKey)")
                    }
                    group.addTask { @Sendable in
                        let localLogitsPart = logitsPart
                        let localOffset = (partIndex - 1) * logitsPart.count
                        
                        guard let pixelBuffer = localLogitsPart.pixelBuffer else {
                            throw InferenceError.inferenceError("Missing or invalid \(logitsKey) in output backings")
                        }
                        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
                        defer {
                            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
                        }
                        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                            throw InferenceError.inferenceError("Could not get base address for \(logitsKey)")
                        }
                        
                        #if arch(arm64)
                        let buffer = baseAddress.assumingMemoryBound(to: Float16.self)
                        let count = localLogitsPart.count
                        var output: [(Int, Float)] = []
                        var start = 0
                        if localOffset == 0 && self.FilterLLAMA01 {
                            start = 2  // filtering special tokens
                            if self.debugLevel >= 2 {
                                print("Filtering special tokens: start=\(_padTokenId)")
                            }
                        }
                        
                        for j in start..<count {
                            let value = Float(buffer[j])
                            let globalIndex = localOffset + j
                            output.append((globalIndex, value))
                        }
                        return output
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
            
            // Flatten results into a single array of (index, logit) pairs.
            let flatLogits = logitsResults.flatMap { $0 }
            
            // Determine the vocabulary size: assume it equals the max index observed + 1.
            let vocabSize = (flatLogits.map { $0.0 }.max() ?? 0) + 1
            var logitsVector = Array(repeating: -Float.infinity, count: vocabSize)
            for (index, value) in flatLogits {
                logitsVector[index] = value
            }
            
            // Use top-p sampling to pick a token. You can hardcode or pass in your topP (here 0.9).
            let sampledIndex = topPSample(logits: logitsVector, temperature: temperature, topP: 0.9)
            if debugLevel >= 1 {
                print("\nTop-p sampled token:", sampledIndex)
            }
            return sampledIndex
        }   
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
        eosToken: Int,
        tokenizer: Tokenizer,
        onToken: ((Int) -> Void)? = nil,
        onWindowShift: (() -> Void)? = nil
    ) async throws -> ([Int], TimeInterval, String) {
        
        var generatedTokens: [Int] = []
        let startTime = CFAbsoluteTimeGetCurrent()
        var stopReason = "max_tokens"
        // Use known EOT token ID
        let eotToken = 128009  // Hardcode the token ID we observed

        if (busy) {
            print("Should not happen!!!!!")
            generatedTokens.append(eotToken)
            return  (generatedTokens, 0, "Inference is Busy")
        }

        let _padTokenId = tokenizer.padTokenId
        abort_generation = 0;
        busy = true
        
        do {

            if debugLevel >= 1 {
                print("\n=== EOT Token Setup ===")
                print("EOT token ID: \(eotToken)")
                print("EOT decoded: '\(tokenizer.decode(tokens: [eotToken], skipSpecialTokens: false))'")
            }
            
            // Create mutable copy of initialTokens
            var contextTokens = initialTokens
            
            // Run prefill with mutable copy
            var currentPos = try await runPrefill(on: &contextTokens, contextPos: contextTokens.count, tokenizer: tokenizer)
            let prefillTime = CFAbsoluteTimeGetCurrent() - startTime
            
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
                
                let nextToken = try await generateNextToken(
                    for: contextTokens[currentPos - 1],
                    currentPos: currentPos,
                    temperature: temperature,
                    tokenizer: tokenizer
                )
                
                // Debug token comparison
                if debugLevel >= 1 {
                    print("\nToken check:")
                    print("Next token: \(nextToken)")
                    print("Decoded: '\(tokenizer.decode(tokens: [nextToken], skipSpecialTokens: false))'")
                    print("Is EOS? \(nextToken == eosToken)")
                    print("Is EOT? \(nextToken == eotToken)")  // Direct comparison
                }
                
                // Check for stop tokens before adding to generated tokens
                //if nextToken == eosToken || nextToken == 0 {
                if nextToken == eosToken  { // cannot be zero ?

                    stopReason = "eos_token"
                    if debugLevel >= 1 {
                        print("\nStopping: EOS token detected (\(nextToken))")
                    }
                    break
                }
                
                if nextToken == eotToken {  // Direct comparison
                    stopReason = "eot_token"
                    if debugLevel >= 1 {
                        print("\nStopping: EOT token detected (\(nextToken))")
                    }
                    break
                }
                
                // Only add token and continue if not a stop token
                generatedTokens.append(nextToken)
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
        hiddenStatesBackings_last = nil
        hiddenStatesBackings_emb_prefill = nil
        hiddenStatesBackings_ffn_prefill = nil
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
