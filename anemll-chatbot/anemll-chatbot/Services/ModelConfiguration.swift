// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// ModelConfiguration.swift

import Foundation

// ModelConfiguration struct for parsing model metadata
public struct ModelConfiguration {
    let modelPrefix: String
    let numChunks: Int
    let lutLMHead: Int?
    let lutFFN: Int?
    let lutEmbeddings: Int?
    let contextLength: Int
    let batchSize: Int
    let version: String
    var modelPath: String?  // Add model path property
    let splitLMHead: Int  // Add split_lm_head parameter for Qwen support
    let modelType: String  // "monolithic" or "chunked" (default)
    let monolithicModel: String?  // Name of monolithic .mlmodelc file (e.g., "gemma3_monolithic_full_lut6.mlmodelc")
    let argmaxInModel: Bool  // If true, model outputs argmax_idx/val pairs instead of logits
    let slidingWindow: Int?  // nil means no rotation needed, 512 for Gemma3

    /// Returns true if this is a monolithic model (single combined model file)
    var isMonolithic: Bool {
        return modelType.lowercased() == "monolithic" && monolithicModel != nil
    }
    
    // Make shouldUseV110 settable while keeping the default behavior
    var shouldUseV110: Bool {
        get {
            return version == "0.1.1" || version.hasPrefix("v0.1.1")
        }
        set {
            // This is a trick to allow setting the property - we'll actually store it in _manualV110Flag
            _manualV110Flag = newValue
        }
    }
    
    // Private backing field for manual override
    private var _manualV110Flag: Bool?
    
    // Allow manual override of v110 flag
    var manuallySetV110: Bool? {
        get { return _manualV110Flag }
        set { _manualV110Flag = newValue }
    }
    
    init(from yamlContent: String, modelPath: String? = nil) throws {
        // Default values
        var modelPrefix = "model"
        var numChunks = 1
        var lutLMHead: Int? = nil
        var lutFFN: Int? = nil
        var lutEmbeddings: Int? = nil
        var contextLength = 2048
        var batchSize = 512
        var version = "0.0.0"
        var splitLMHead = 8  // Default to 8 for LLaMA models
        var modelType = "chunked"  // Default to chunked (separate embed/FFN/lmhead)
        var monolithicModel: String? = nil  // Only set for monolithic models
        var argmaxInModel = false  // Default to false (model outputs logits)
        var slidingWindow: Int? = nil  // Default to nil (no rotation needed)
        
        // Helper function to extract parameter value from a section
        func extractParameterFromSection(section: String, key: String) -> String? {
            let lines = section.components(separatedBy: .newlines)
            for line in lines {
                let trimmedLine = line.trimmingCharacters(in: .whitespaces)
                if trimmedLine.hasPrefix("\(key):") {
                    let valueParts = trimmedLine.components(separatedBy: ":")
                    if valueParts.count >= 2 {
                        let value = valueParts[1].trimmingCharacters(in: .whitespacesAndNewlines)
                        return value
                    }
                }
            }
            return nil
        }
        
        // Helper function to parse LUT value that might be boolean or integer
        func parseLutValue(_ value: String) -> Int? {
            // Check if it's a boolean first
            if value.lowercased() == "true" {
                return 1
            } else if value.lowercased() == "false" {
                return 0
            }
            // Otherwise try to parse as integer
            return Int(value)
        }
        
        // First, try to find the model_info section
        let modelInfoPattern = "model_info:"
        if let modelInfoRange = yamlContent.range(of: modelInfoPattern) {
            // Extract the model_info section
            let modelInfoStart = modelInfoRange.upperBound
            let modelInfoContent = String(yamlContent[modelInfoStart...])
            
            // Look for version in model_info section
            if let versionLine = modelInfoContent.components(separatedBy: .newlines).first(where: { $0.contains("version:") }) {
                let versionParts = versionLine.components(separatedBy: "version:")
                if versionParts.count >= 2 {
                    version = versionParts[1].trimmingCharacters(in: .whitespacesAndNewlines)
                    print("✅ Found version in model_info: \(version)")
                }
            }
            
            // Look for nested parameters section within model_info
            let nestedParamsPattern = "  parameters:"
            if let nestedParamsRange = modelInfoContent.range(of: nestedParamsPattern) {
                // Extract the nested parameters section
                let nestedParamsStart = nestedParamsRange.upperBound
                let nestedParamsContent = String(modelInfoContent[nestedParamsStart...])
                
                // Parse model_prefix from nested parameters
                if let prefixValue = extractParameterFromSection(section: nestedParamsContent, key: "model_prefix") {
                    print("🔍 Parsing model_prefix from model_info.parameters: '\(prefixValue)'")
                    modelPrefix = prefixValue
                    print("✅ Set modelPrefix to \(modelPrefix)")
                }
                
                // Parse num_chunks from nested parameters
                if let chunksValue = extractParameterFromSection(section: nestedParamsContent, key: "num_chunks") {
                    print("🔍 Parsing num_chunks from model_info.parameters: '\(chunksValue)'")
                    if let chunksInt = Int(chunksValue) {
                        numChunks = chunksInt
                        print("✅ Set numChunks to \(chunksInt)")
                    }
                }
                
                // Parse lut_lmhead from nested parameters
                if let lutValue = extractParameterFromSection(section: nestedParamsContent, key: "lut_lmhead") {
                    print("🔍 Parsing lut_lmhead from model_info.parameters: '\(lutValue)'")
                    if let parsedValue = parseLutValue(lutValue) {
                        lutLMHead = parsedValue
                        print("✅ Set lutLMHead to \(parsedValue)")
                    }
                }
                
                // Parse lut_ffn from nested parameters
                if let lutValue = extractParameterFromSection(section: nestedParamsContent, key: "lut_ffn") {
                    print("🔍 Parsing lut_ffn from model_info.parameters: '\(lutValue)'")
                    if let parsedValue = parseLutValue(lutValue) {
                        lutFFN = parsedValue
                        print("✅ Set lutFFN to \(parsedValue)")
                    }
                }
                
                // Parse lut_embeddings from nested parameters
                if let lutValue = extractParameterFromSection(section: nestedParamsContent, key: "lut_embeddings") {
                    print("🔍 Parsing lut_embeddings from model_info.parameters: '\(lutValue)'")
                    if let parsedValue = parseLutValue(lutValue) {
                        lutEmbeddings = parsedValue
                        print("✅ Set lutEmbeddings to \(parsedValue)")
                    }
                }
                
                // Parse context_length from nested parameters
                if let contextValue = extractParameterFromSection(section: nestedParamsContent, key: "context_length") {
                    print("🔍 Parsing context_length from model_info.parameters: '\(contextValue)'")
                    if let contextInt = Int(contextValue) {
                        contextLength = contextInt
                        print("✅ Set contextLength to \(contextInt)")
                    }
                }
                
                // Parse batch_size from nested parameters
                if let batchValue = extractParameterFromSection(section: nestedParamsContent, key: "batch_size") {
                    print("🔍 Parsing batch_size from model_info.parameters: '\(batchValue)'")
                    if let batchInt = Int(batchValue) {
                        batchSize = batchInt
                        print("✅ Set batchSize to \(batchInt)")
                    }
                }
                
                // Parse split_lm_head from nested parameters
                if let splitValue = extractParameterFromSection(section: nestedParamsContent, key: "split_lm_head") {
                    print("🔍 Parsing split_lm_head from model_info.parameters: '\(splitValue)'")
                    if let splitInt = Int(splitValue) {
                        splitLMHead = splitInt
                        print("✅ Set splitLMHead to \(splitInt)")
                    }
                }

                // Parse monolithic_model from nested parameters (for monolithic models)
                if let monolithicValue = extractParameterFromSection(section: nestedParamsContent, key: "monolithic_model") {
                    print("🔍 Parsing monolithic_model from model_info.parameters: '\(monolithicValue)'")
                    monolithicModel = monolithicValue
                    print("✅ Set monolithicModel to \(monolithicValue)")
                }

                // Parse argmax_in_model from nested parameters (for models with built-in argmax)
                if let argmaxValue = extractParameterFromSection(section: nestedParamsContent, key: "argmax_in_model") {
                    print("🔍 Parsing argmax_in_model from model_info.parameters: '\(argmaxValue)'")
                    argmaxInModel = argmaxValue.lowercased() == "true"
                    print("✅ Set argmaxInModel to \(argmaxInModel)")
                }

                // Parse sliding_window from nested parameters (for Gemma3 rotation support)
                if let slidingValue = extractParameterFromSection(section: nestedParamsContent, key: "sliding_window") {
                    print("🔍 Parsing sliding_window from model_info.parameters: '\(slidingValue)'")
                    if let slidingInt = Int(slidingValue) {
                        slidingWindow = slidingInt
                        print("✅ Set slidingWindow to \(slidingInt)")
                    }
                }
            } else {
                print("⚠️ No nested parameters section found in model_info section")
            }

            // Look for model_type in model_info section (outside of parameters)
            if let typeValue = extractParameterFromSection(section: modelInfoContent, key: "model_type") {
                print("🔍 Parsing model_type from model_info: '\(typeValue)'")
                modelType = typeValue
                print("✅ Set modelType to \(typeValue)")
            }
        } else {
            print("⚠️ No model_info section found in YAML")
        }
        
        // Also check for batch_size at root level (older models)
        if let batchValue = extractParameterFromSection(section: yamlContent, key: "batch_size") {
            print("🔍 Parsing batch_size from root level: '\(batchValue)'")
            if let batchInt = Int(batchValue) {
                batchSize = batchInt
                print("✅ Set batchSize to \(batchInt) from root level")
            }
        } else {
            print("⚠️ No batch_size field found at root level, using current value: \(batchSize)")
        }
        
        self.modelPrefix = modelPrefix
        self.numChunks = numChunks
        self.lutLMHead = lutLMHead
        self.lutFFN = lutFFN
        self.lutEmbeddings = lutEmbeddings
        self.contextLength = contextLength
        self.batchSize = batchSize
        self.version = version
        self.modelPath = modelPath
        self.splitLMHead = splitLMHead
        self.modelType = modelType
        self.monolithicModel = monolithicModel
        self.argmaxInModel = argmaxInModel
        self.slidingWindow = slidingWindow

        print("📊 ModelConfiguration initialized: modelPrefix='\(modelPrefix)', numChunks=\(numChunks), lutLMHead=\(String(describing: lutLMHead)), lutFFN=\(String(describing: lutFFN)), lutEmbeddings=\(String(describing: lutEmbeddings)), contextLength=\(contextLength), batchSize=\(batchSize), splitLMHead=\(splitLMHead), version=\(version), modelType=\(modelType), monolithicModel=\(String(describing: monolithicModel)), argmaxInModel=\(argmaxInModel), slidingWindow=\(String(describing: slidingWindow))")
        
        let v110Reason = self._manualV110Flag != nil ? "MANUALLY SET" : "version check"
        print("📊 v110 flag is: \(self.shouldUseV110 ? "TRUE" : "FALSE") (\(v110Reason)) based on version \(version)")
        if let path = self.modelPath {
            print("📂 Model path: \(path)")
        }
    }
} 