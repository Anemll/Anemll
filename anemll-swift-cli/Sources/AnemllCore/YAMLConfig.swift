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
            var missingFiles: [String] = []
            for path in [embedPath, lmheadPath, ffnPath] {
                if !fileManager.fileExists(atPath: path) {
                    print("Error: Model file not found: \(path)")
                    missingFiles.append(path)
                    
                    // Try to list similar files to help with debugging
                    let directory = (path as NSString).deletingLastPathComponent
                    if let files = try? fileManager.contentsOfDirectory(atPath: directory) {
                        print("Files in directory \(directory):")
                        for file in files {
                            print("  - \(file)")
                        }
                    }
                }
            }
            
            if !missingFiles.isEmpty {
                throw ConfigError.missingField("Model files not found: \(missingFiles.joined(separator: ", "))")
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
        } catch let yamlError as ConfigError {
            // Re-throw ConfigError
            throw yamlError
        } catch {
            print("Error parsing YAML: \(error.localizedDescription)")
            throw ConfigError.invalidFormat("Failed to parse YAML: \(error.localizedDescription)")
        }
    }
}

public enum ConfigError: Error {
    case invalidFormat(String)
    case missingField(String)
} 