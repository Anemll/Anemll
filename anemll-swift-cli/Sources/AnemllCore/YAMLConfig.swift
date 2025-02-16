import Foundation
import Yams

public struct YAMLConfig {
    public let modelPath: String
    public let configVersion: String
    public let functionName: String?
    public let tokenizerModel: String
    public let contextLength: Int
    public let stateLength: Int
    public let batchSize: Int
    public let lutBits: Int
    public let numChunks: Int
    
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
        
        // Extract paths from yaml
        self.embedPath = yaml["embed_path"] as? String ?? ""
        self.lmheadPath = yaml["lmhead_path"] as? String ?? ""
        self.ffnPath = yaml["ffn_path"] as? String ?? ""
        
        self.configVersion = yaml["version"] as? String ?? "1.0"
    }
    
    /// Load configuration from a file path
    public static func load(from path: String) throws -> YAMLConfig {
        print("Reading YAML from: \(path)")
        let configString = try String(contentsOfFile: path, encoding: .utf8)
        print("YAML contents: \(configString)")
        
        // Parse YAML
        guard let yaml = try Yams.load(yaml: configString) as? [String: Any],
              let modelInfo = yaml["model_info"] as? [String: Any],
              let params = modelInfo["parameters"] as? [String: Any] else {
            throw ConfigError.invalidFormat("Failed to parse YAML structure")
        }
        
        // Get directory containing meta.yaml
        let baseDir = (path as NSString).deletingLastPathComponent
        
        // Extract parameters from modelInfo["parameters"]
        let modelPrefix = params["model_prefix"] as? String ?? "llama"
        let lutFFN = String(params["lut_ffn"] as? Int ?? -1)
        let lutLMHead = String(params["lut_lmhead"] as? Int ?? -1)
        let numChunks = params["num_chunks"] as? Int ?? 1
        
        // Build paths
        let embedPath = "\(baseDir)/\(modelPrefix)_embeddings.mlmodelc"
        let lmheadPath = "\(baseDir)/\(modelPrefix)_lm_head\(lutLMHead != "-1" ? "_lut\(lutLMHead)" : "").mlmodelc"
        let ffnPath = "\(baseDir)/\(modelPrefix)_FFN_PF\(lutFFN != "-1" ? "_lut\(lutFFN)" : "")_chunk_01of\(String(format: "%02d", numChunks)).mlmodelc"
        
        print("\nModel paths (Python style):")
        print("Raw paths before .mlmodelc:")
        print("Embed: \(modelPrefix)_embeddings")
        print("LMHead: \(modelPrefix)_lm_head\(lutLMHead != "none" ? "_lut\(lutLMHead)" : "")")
        print("FFN: \(modelPrefix)_FFN_PF\(lutFFN != "none" ? "_lut\(lutFFN)" : "")_chunk_01of\(String(format: "%02d", numChunks))")
        print("\nFull paths:")
        print("Embed: \(embedPath)")
        print("LMHead: \(lmheadPath)")
        print("FFN: \(ffnPath)")
        
        // Verify files exist
        let fileManager = FileManager.default
        for path in [embedPath, lmheadPath, ffnPath] {
            guard fileManager.fileExists(atPath: path) else {
                print("Error: Model file not found: \(path)")
                throw ConfigError.missingField("Model file not found: \(path)")
            }
        }
        
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
            "version": modelInfo["version"] as? String ?? "1.0",
            "embed_path": embedPath,
            "ffn_path": ffnPath,
            "lmhead_path": lmheadPath
        ]
        
        let yamlString = try Yams.dump(object: configDict)
        return try YAMLConfig(from: yamlString)
    }
}

public enum ConfigError: Error {
    case invalidFormat(String)
    case missingField(String)
} 