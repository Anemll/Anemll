import Foundation
import Yams

public struct YAMLConfig: Sendable {
    public struct RecommendedSampling: Sendable {
        public let doSample: Bool
        public let temperature: Double
        public let topK: Int
        public let topP: Double
    }

    public let modelPath: String
    public let configVersion: String
    public let functionName: String?
    public let tokenizerModel: String
    public let contextLength: Int
    public let stateLength: Int
    public let batchSize: Int
    public let lutBits: Int
    public let numChunks: Int
    public let splitLMHead: Int
    public let vocabSize: Int?
    public let lmHeadChunkSizes: [Int]?

    // Model paths
    public let embedPath: String
    public let ffnPath: String
    public let lmheadPath: String

    // Monolithic model support
    public let isMonolithic: Bool
    public let monolithicModelPath: String?
    public let argmaxInModel: Bool  // If true, model outputs argmax_idx/val pairs instead of logits

    // Gemma3 sliding window support (for 4-function models with rotation)
    public let slidingWindow: Int?  // nil means no rotation needed, 512 for Gemma3

    // Update mask support for prefill (for proper multi-turn KV cache handling)
    public let updateMaskPrefill: Bool  // If true, prefill expects update_mask input

    // Dynamic slice prefill support (alternative to update_mask for batch prefill)
    public let prefillDynamicSlice: Bool  // If true, model supports dynamic slice prefill

    // Model prefix for template auto-detection
    public let modelPrefix: String

    // Optional model-level sampling recommendation (from meta.yaml)
    public let recommendedSampling: RecommendedSampling?
    
    public init(from yamlString: String) throws {
        // Load YAML
        guard let yaml = try Yams.load(yaml: yamlString) as? [String: Any] else {
            throw ConfigError.invalidFormat("Failed to parse YAML")
        }
        
        // Extract required fields
        guard let modelPath = yaml["model_path"] as? String else {
            throw ConfigError.missingField("model_path")
        }
        guard let tokenizerModel = yaml["tokenizer_model"] as? String else {
            throw ConfigError.missingField("tokenizer_model")
        }
        
        // Extract optional fields with defaults
        self.contextLength = yaml["context_length"] as? Int ?? 2048
        self.batchSize = yaml["batch_size"] as? Int ?? 32
        self.functionName = yaml["function_name"] as? String
        
        self.modelPath = modelPath
        self.tokenizerModel = tokenizerModel
        
        // Extract model parameters
        self.stateLength = yaml["state_length"] as? Int ?? self.contextLength
        self.lutBits = yaml["lut_bits"] as? Int ?? 4
        self.numChunks = yaml["num_chunks"] as? Int ?? 1
        self.splitLMHead = yaml["split_lm_head"] as? Int ?? 8
        self.vocabSize = yaml["vocab_size"] as? Int
        self.lmHeadChunkSizes = yaml["lm_head_chunk_sizes"] as? [Int]
        
        // Extract paths from yaml
        self.embedPath = yaml["embed_path"] as? String ?? ""
        self.lmheadPath = yaml["lmhead_path"] as? String ?? ""

        // Monolithic model support
        self.isMonolithic = yaml["is_monolithic"] as? Bool ?? false
        self.monolithicModelPath = yaml["monolithic_model_path"] as? String
        self.argmaxInModel = yaml["argmax_in_model"] as? Bool ?? false

        // Sliding window support for Gemma3
        self.slidingWindow = yaml["sliding_window"] as? Int

        // Update mask prefill support
        self.updateMaskPrefill = yaml["update_mask_prefill"] as? Bool ?? false

        // Dynamic slice prefill support
        self.prefillDynamicSlice = yaml["prefill_dynamic_slice"] as? Bool ?? false

        // Model prefix for template auto-detection
        self.modelPrefix = yaml["model_prefix"] as? String ?? "llama"

        // Optional recommended sampling block
        if let sampling = yaml["recommended_sampling"] as? [String: Any],
           let temperature = YAMLConfig.toDouble(sampling["temperature"]),
           let topP = YAMLConfig.toDouble(sampling["top_p"] ?? sampling["topP"]),
           let topK = YAMLConfig.toInt(sampling["top_k"] ?? sampling["topK"]) {
            let doSample = sampling["do_sample"] as? Bool ?? true
            self.recommendedSampling = RecommendedSampling(
                doSample: doSample,
                temperature: temperature,
                topK: topK,
                topP: topP
            )
        } else {
            self.recommendedSampling = nil
        }

        // Get the ffn_path
        let rawFFNPath = yaml["ffn_path"] as? String ?? ""

        // If multi-chunk model and path doesn't already have the proper format, adjust it
        if self.numChunks > 1 && !rawFFNPath.contains("_chunk_01of") {
            let directory = (rawFFNPath as NSString).deletingLastPathComponent
            let filename = (rawFFNPath as NSString).lastPathComponent

            // Derive base name without .mlmodelc
            var baseName = filename
            if baseName.hasSuffix(".mlmodelc") {
                baseName = String(baseName.dropLast(9)) // Remove .mlmodelc
            }

            // Generate canonical first chunk path
            self.ffnPath = "\(directory)/\(baseName)_chunk_01of\(String(format: "%02d", self.numChunks)).mlmodelc"
            print("Generated canonical chunk path: \(self.ffnPath)")
        } else {
            self.ffnPath = rawFFNPath
        }

        self.configVersion = yaml["version"] as? String ?? "0.3.5"
    }
    
    /// Load configuration from a file path
    public static func load(from path: String) throws -> YAMLConfig {
        print("Reading YAML from: \(path)")
        
        // Check if the file exists
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: path) else {
            print("Error: YAML file not found at path: \(path)")
            throw ConfigError.invalidFormat("YAML file not found at path: \(path)")
        }
        
        // Read the file contents
        let configString: String
        do {
            configString = try String(contentsOfFile: path, encoding: .utf8)
            print("YAML contents loaded successfully")
        } catch {
            print("Error reading YAML file: \(error.localizedDescription)")
            throw ConfigError.invalidFormat("Failed to read YAML file: \(error.localizedDescription)")
        }
        
        // Parse YAML
        do {
            guard let yaml = try Yams.load(yaml: configString) as? [String: Any] else {
                print("Error: YAML content could not be parsed as dictionary")
                throw ConfigError.invalidFormat("YAML content could not be parsed as dictionary")
            }
            
            guard let modelInfo = yaml["model_info"] as? [String: Any] else {
                print("Error: Missing 'model_info' section in YAML")
                throw ConfigError.missingField("model_info")
            }
            
            guard let params = modelInfo["parameters"] as? [String: Any] else {
                print("Error: Missing 'parameters' section in model_info")
                throw ConfigError.missingField("model_info.parameters")
            }
            
            // Get directory containing meta.yaml
            let baseDir = (path as NSString).deletingLastPathComponent
            print("Base directory: \(baseDir)")
            
            // Extract parameters from modelInfo["parameters"]
            let modelPrefix = params["model_prefix"] as? String ?? "llama"
            print("Model prefix: \(modelPrefix)")

            // Detect monolithic model type
            let modelType = modelInfo["model_type"] as? String ?? "chunked"
            let isMonolithic = (modelType == "monolithic")
            print("Model type: \(modelType), isMonolithic: \(isMonolithic)")

            let lutFFN = String(params["lut_ffn"] as? Int ?? -1)
            let lutLMHead = String(params["lut_lmhead"] as? Int ?? -1)
            let lutEmbeddings = String(params["lut_embeddings"] as? Int ?? -1)
            let numChunks = params["num_chunks"] as? Int ?? 1
            let splitLMHead = params["split_lm_head"] as? Int ?? 8
            let vocabSize = params["vocab_size"] as? Int
            let lmHeadChunkSizes = params["lm_head_chunk_sizes"] as? [Int]
            let recommendedSampling = params["recommended_sampling"] as? [String: Any]
            
            // Check for predefined paths in parameters
            let predefinedEmbedPath = params["embeddings"] as? String
            let predefinedLMHeadPath = params["lm_head"] as? String
            let predefinedFFNPath = params["ffn"] as? String
            let predefinedMonolithicModel = params["monolithic_model"] as? String

            print("Predefined paths from meta.yaml:")
            print("  - embeddings: \(predefinedEmbedPath ?? "Not defined")")
            print("  - lm_head: \(predefinedLMHeadPath ?? "Not defined")")
            print("  - ffn: \(predefinedFFNPath ?? "Not defined")")
            print("  - monolithic_model: \(predefinedMonolithicModel ?? "Not defined")")
            
            // Build paths, preferring predefined paths if available
            let embedPath: String
            if let definedPath = predefinedEmbedPath {
                embedPath = "\(baseDir)/\(definedPath)"
            } else {
                // Always include "_embeddings" suffix with optional LUT suffix
                embedPath = "\(baseDir)/\(modelPrefix)_embeddings\(lutEmbeddings != "-1" ? "_lut\(lutEmbeddings)" : "").mlmodelc"
            }
            
            let lmheadPath: String
            if let definedPath = predefinedLMHeadPath {
                lmheadPath = "\(baseDir)/\(definedPath)"
            } else {
                lmheadPath = "\(baseDir)/\(modelPrefix)_lm_head\(lutLMHead != "-1" ? "_lut\(lutLMHead)" : "").mlmodelc"
            }
            
            let ffnPath: String
            if let definedPath = predefinedFFNPath {
                // Check if the predefined path exists, or if we need to add chunk suffix
                let fullPath = "\(baseDir)/\(definedPath)"
                if FileManager.default.fileExists(atPath: fullPath) {
                    ffnPath = fullPath
                } else if numChunks == 1 {
                    // Try with _chunk_01of01 suffix
                    let pathWithoutExt = definedPath.replacingOccurrences(of: ".mlmodelc", with: "")
                    let chunkedPath = "\(baseDir)/\(pathWithoutExt)_chunk_01of01.mlmodelc"
                    if FileManager.default.fileExists(atPath: chunkedPath) {
                        ffnPath = chunkedPath
                        print("Found single-chunk FFN model with chunk suffix: \(chunkedPath)")
                    } else {
                        ffnPath = fullPath // Fall back to original path
                    }
                } else {
                    ffnPath = fullPath
                }
            } else if numChunks > 1 {
                // For multi-chunk models, use the canonical chunk path format
                ffnPath = "\(baseDir)/\(modelPrefix)_FFN_PF\(lutFFN != "-1" ? "_lut\(lutFFN)" : "")_chunk_01of\(String(format: "%02d", numChunks)).mlmodelc"
                print("Generated canonical chunked FFN path: \(ffnPath)")
            } else {
                // For single-chunk models, check if _chunk_01of01 exists
                let baseFFNPath = "\(baseDir)/\(modelPrefix)_FFN_PF\(lutFFN != "-1" ? "_lut\(lutFFN)" : "")"
                let chunkedPath = "\(baseFFNPath)_chunk_01of01.mlmodelc"
                let nonChunkedPath = "\(baseFFNPath).mlmodelc"
                
                // Check if chunked version exists
                if FileManager.default.fileExists(atPath: chunkedPath) {
                    ffnPath = chunkedPath
                    print("Found single-chunk FFN model with chunk suffix: \(chunkedPath)")
                } else {
                    ffnPath = nonChunkedPath
                }
            }
            
            // Check for argmax_in_model flag
            let argmaxInModel = params["argmax_in_model"] as? Bool ?? false

            // Check for sliding_window (Gemma3 rotation support)
            // If sliding_window is explicitly set in params, use it
            // Otherwise, detect Gemma3 prefix and use 512 if context > 512
            let slidingWindow: Int?
            if let sw = params["sliding_window"] as? Int {
                slidingWindow = sw
                print("Sliding window from meta.yaml: \(sw)")
            } else if modelPrefix.lowercased().hasPrefix("gemma3") {
                let contextLength = params["context_length"] as? Int ?? 2048
                if contextLength > 512 {
                    slidingWindow = 512
                    print("Gemma3 detected with context > 512, defaulting sliding_window to 512")
                } else {
                    slidingWindow = nil
                }
            } else {
                slidingWindow = nil
            }

            // Check for update_mask_prefill flag (for proper multi-turn KV cache handling)
            let updateMaskPrefill = params["update_mask_prefill"] as? Bool ?? false
            if updateMaskPrefill {
                print("Update mask prefill enabled")
            }

            // Check for prefill_dynamic_slice flag (alternative batch prefill support)
            let prefillDynamicSlice = params["prefill_dynamic_slice"] as? Bool ?? false
            if prefillDynamicSlice {
                print("Prefill dynamic slice enabled")
            }

            if let rec = recommendedSampling,
               let recTemp = YAMLConfig.toDouble(rec["temperature"]),
               let recTopP = YAMLConfig.toDouble(rec["top_p"] ?? rec["topP"]),
               let recTopK = YAMLConfig.toInt(rec["top_k"] ?? rec["topK"]) {
                let recDoSample = rec["do_sample"] as? Bool ?? true
                print("Recommended sampling from meta.yaml: do_sample=\(recDoSample), temperature=\(recTemp), top_p=\(recTopP), top_k=\(recTopK)")
            }

            // Build monolithic model path if applicable
            let monolithicModelPath: String?
            if isMonolithic {
                if let definedPath = predefinedMonolithicModel {
                    monolithicModelPath = "\(baseDir)/\(definedPath)"
                } else {
                    // Fallback to constructing the path
                    let lutSuffix = (params["lut_bits"] as? Int).map { "_lut\($0)" } ?? ""
                    monolithicModelPath = "\(baseDir)/\(modelPrefix)_monolithic_full\(lutSuffix).mlmodelc"
                }
                print("\nMonolithic model path: \(monolithicModelPath!)")
                print("Argmax in model: \(argmaxInModel)")
            } else {
                monolithicModelPath = nil
                print("\nModel paths (Python style):")
                print("Raw paths before .mlmodelc:")
                print("Embed: \(modelPrefix)_embeddings\(lutEmbeddings != "-1" ? "_lut\(lutEmbeddings)" : "")")
                print("LMHead: \(modelPrefix)_lm_head\(lutLMHead != "-1" ? "_lut\(lutLMHead)" : "")")
                if numChunks > 1 {
                    print("FFN: \(modelPrefix)_FFN_PF\(lutFFN != "-1" ? "_lut\(lutFFN)" : "")_chunk_01of\(String(format: "%02d", numChunks))")
                } else {
                    print("FFN: \(modelPrefix)_FFN_PF\(lutFFN != "-1" ? "_lut\(lutFFN)" : "")")
                }
                print("\nFull paths:")
                print("Embed: \(embedPath)")
                print("LMHead: \(lmheadPath)")
                print("FFN: \(ffnPath)")
            }
            
            // Create YAML string for init(from:)
            var configDict: [String: Any] = [
                "model_path": isMonolithic ? (monolithicModelPath ?? "") : ffnPath,
                "tokenizer_model": baseDir,
                "context_length": params["context_length"] as? Int ?? 2048,
                "batch_size": params["batch_size"] as? Int ?? 32,
                "state_length": params["context_length"] as? Int ?? 2048,
                "lut_bits": params["lut_bits"] as? Int ?? 4,
                "num_chunks": numChunks,
                "model_prefix": modelPrefix,
                "lut_ffn": lutFFN,
                "lut_lmhead": lutLMHead,
                "lut_embeddings": lutEmbeddings,
                "version": modelInfo["version"] as? String ?? "1.0",
                "embed_path": embedPath,
                "ffn_path": ffnPath,
                "lmhead_path": lmheadPath,
                "split_lm_head": splitLMHead,
                "vocab_size": vocabSize as Any,
                "lm_head_chunk_sizes": lmHeadChunkSizes as Any,
                "is_monolithic": isMonolithic,
                "argmax_in_model": argmaxInModel,
                "update_mask_prefill": updateMaskPrefill,
                "prefill_dynamic_slice": prefillDynamicSlice
            ]
            if let rec = recommendedSampling {
                configDict["recommended_sampling"] = rec
            }
            if let monolithicPath = monolithicModelPath {
                configDict["monolithic_model_path"] = monolithicPath
            }
            if let sw = slidingWindow {
                configDict["sliding_window"] = sw
            }
            
            let yamlString = try Yams.dump(object: configDict)
            return try YAMLConfig(from: yamlString)
        } catch let yamlError as ConfigError {
            // Re-throw ConfigError
            throw yamlError
        } catch {
            print("Error parsing YAML: \(error.localizedDescription)")
            throw ConfigError.invalidFormat("Failed to parse YAML: \(error.localizedDescription)")
        }
    }
    
    // Helper method to create YAMLConfig when an alternate chunk is found
    private static func loadFromDetectedPaths(
        baseDir: String,
        embedPath: String,
        lmheadPath: String,
        ffnPath: String,
        params: [String: Any],
        modelInfo: [String: Any],
        modelPrefix: String,
        numChunks: Int,
        lutFFN: String,
        lutLMHead: String,
        lutEmbeddings: String,
        splitLMHead: Int,
        isMonolithic: Bool = false,
        monolithicModelPath: String? = nil,
        argmaxInModel: Bool = false,
        slidingWindow: Int? = nil,
        recommendedSampling: [String: Any]? = nil
    ) throws -> YAMLConfig {
        // Create YAML string for init(from:)
        var configDict: [String: Any] = [
            "model_path": isMonolithic ? (monolithicModelPath ?? "") : ffnPath,
            "tokenizer_model": baseDir,
            "context_length": params["context_length"] as? Int ?? 2048,
            "batch_size": params["batch_size"] as? Int ?? 32,
            "state_length": params["context_length"] as? Int ?? 2048,
            "lut_bits": params["lut_bits"] as? Int ?? 4,
            "num_chunks": numChunks,
            "model_prefix": modelPrefix,
            "lut_ffn": lutFFN,
            "lut_lmhead": lutLMHead,
            "lut_embeddings": lutEmbeddings,
            "version": modelInfo["version"] as? String ?? "1.0",
            "embed_path": embedPath,
            "ffn_path": ffnPath,
            "lmhead_path": lmheadPath,
            "split_lm_head": splitLMHead,
            "vocab_size": params["vocab_size"] as? Int as Any,
            "lm_head_chunk_sizes": params["lm_head_chunk_sizes"] as? [Int] as Any,
            "is_monolithic": isMonolithic,
            "argmax_in_model": argmaxInModel
        ]
        if let rec = recommendedSampling {
            configDict["recommended_sampling"] = rec
        }
        if let monolithicPath = monolithicModelPath {
            configDict["monolithic_model_path"] = monolithicPath
        }
        if let sw = slidingWindow {
            configDict["sliding_window"] = sw
        }

        let yamlString = try Yams.dump(object: configDict)
        return try YAMLConfig(from: yamlString)
    }

    private static func toDouble(_ value: Any?) -> Double? {
        if let v = value as? Double {
            return v
        }
        if let v = value as? Float {
            return Double(v)
        }
        if let v = value as? Int {
            return Double(v)
        }
        if let v = value as? NSNumber {
            return v.doubleValue
        }
        if let v = value as? String {
            return Double(v)
        }
        return nil
    }

    private static func toInt(_ value: Any?) -> Int? {
        if let v = value as? Int {
            return v
        }
        if let v = value as? NSNumber {
            return v.intValue
        }
        if let v = value as? Double {
            return Int(v)
        }
        if let v = value as? String {
            return Int(v)
        }
        return nil
    }
}

public enum ConfigError: Error {
    case invalidFormat(String)
    case missingField(String)
} 
