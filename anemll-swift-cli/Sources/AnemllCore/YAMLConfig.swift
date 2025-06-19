import Foundation
import Yams

public struct YAMLConfig: Sendable {
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
    
    // Model paths
    public let embedPath: String
    public let ffnPath: String
    public let lmheadPath: String
    
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
        
        // Extract paths from yaml
        self.embedPath = yaml["embed_path"] as? String ?? ""
        self.lmheadPath = yaml["lmhead_path"] as? String ?? ""
        
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
        
        self.configVersion = yaml["version"] as? String ?? "0.3.3"
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
            
            let lutFFN = String(params["lut_ffn"] as? Int ?? -1)
            let lutLMHead = String(params["lut_lmhead"] as? Int ?? -1)
            let lutEmbeddings = String(params["lut_embeddings"] as? Int ?? -1)
            let numChunks = params["num_chunks"] as? Int ?? 1
            let splitLMHead = params["split_lm_head"] as? Int ?? 8
            
            // Check for predefined paths in parameters
            let predefinedEmbedPath = params["embeddings"] as? String
            let predefinedLMHeadPath = params["lm_head"] as? String
            let predefinedFFNPath = params["ffn"] as? String
            
            print("Predefined paths from meta.yaml:")
            print("  - embeddings: \(predefinedEmbedPath ?? "Not defined")")
            print("  - lm_head: \(predefinedLMHeadPath ?? "Not defined")")
            print("  - ffn: \(predefinedFFNPath ?? "Not defined")")
            
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
            
            // Create YAML string for init(from:)
            let configDict: [String: Any] = [
                "model_path": ffnPath,
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
                "split_lm_head": splitLMHead
            ]
            
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
        splitLMHead: Int
    ) throws -> YAMLConfig {
        // Create YAML string for init(from:)
        let configDict: [String: Any] = [
            "model_path": ffnPath,
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
            "split_lm_head": splitLMHead
        ]
        
        let yamlString = try Yams.dump(object: configDict)
        return try YAMLConfig(from: yamlString)
    }
}

public enum ConfigError: Error {
    case invalidFormat(String)
    case missingField(String)
} 