// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// ModelService+ModelsJSON.swift - JSON file handling for model persistence

import Foundation
#if os(macOS)
import AppKit
#else
import UIKit
#endif

// Public extension containing functions for handling models.json
extension ModelService {
    
    /// Path to the models.json file in the documents directory
    var modelsJSONPath: URL {
        #if os(macOS) || targetEnvironment(macCatalyst)
        return modelStorageDirectory.appendingPathComponent("models.json")
        #else
        let docDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docDirectory.appendingPathComponent("models.json")
        #endif
    }
    
    /// Ensure the models.json file exists
    func ensureModelsJSON() {
        // Prevent recursive calls
        guard !ModelService.isEnsuringJSON else {
            print("⚠️ Avoiding recursive call to ensureModelsJSON")
            return
        }
        
        // Set the flag
        ModelService.isEnsuringJSON = true
        defer { ModelService.isEnsuringJSON = false }
        
        // Check if file already exists
        if FileManager.default.fileExists(atPath: modelsJSONPath.path) {
            print("✅ models.json exists at \(modelsJSONPath.path)")
            return
        }
        
        // Create empty models.json if it doesn't exist
        createEmptyModelsJSON()
    }
    
    /// Create an empty models.json file
    private func createEmptyModelsJSON() {
        // Create an empty array for models.json
        let emptyModels: [[String: Any]] = []
        
        do {
            // Convert to JSON data
            let jsonData = try JSONSerialization.data(withJSONObject: emptyModels, options: .prettyPrinted)
            
            // Write to file
            try jsonData.write(to: modelsJSONPath)
            print("✅ Created empty models.json at \(modelsJSONPath.path)")
        } catch {
            print("❌ Error creating models.json: \(error)")
        }
    }
    
    // Add flags to prevent recursive calls
    private static var isSavingModels = false
    private static var isLoadingModels = false
    private static var isEnsuringJSON = false
    internal static var isAddingCustomModel = false
    private static var isUpdatingUserDefaults = false
    private static var isInitializing = true // Set to true for startup
    
    /// Reset the initial state once startup is complete
    func completeJSONInitialization() {
        // Call this at the end of startup sequence
        ModelService.isInitializing = false
        print("✅ JSON initialization completed")
    }
    
    /// Save models to models.json file
    func saveModelsToJSON() {
        // Prevent recursive calls
        guard !ModelService.isSavingModels else {
            print("⚠️ Avoiding recursive call to saveModelsToJSON")
            return
        }
        
        // Set the flag
        ModelService.isSavingModels = true
        defer { ModelService.isSavingModels = false }
        
        // Filter for custom models (those not built-in)
        let customModels = availableModels.filter { $0.id != "llama-3.2-1b" }
        
        print("📝 Preparing to save \(customModels.count) models to models.json")
        
        // Create dictionary structure for the JSON
        let modelsData: [[String: Any]] = customModels.map { model -> [String: Any] in
            var modelDict: [String: Any] = [
                "id": model.id,
                "name": model.name,
                "description": model.description,
                "size": model.size,
                "downloadURL": model.downloadURL,
                "isDownloaded": model.isDownloaded
            ]
            
            // If there's a source URL in the meta.yaml file, include it
            if let sourceURL = getSourceURLFromMetaYaml(for: model.id) {
                modelDict["sourceURL"] = sourceURL
            }
            
            return modelDict
        }
        
        do {
            // Convert to JSON data
            let jsonData = try JSONSerialization.data(withJSONObject: modelsData, options: .prettyPrinted)
            
            // Write to file
            try jsonData.write(to: modelsJSONPath)
            print("✅ Saved \(customModels.count) models to \(modelsJSONPath.path)")
            
            // For backward compatibility, update UserDefaults as well
            // Only do this if we're not in the middle of any sensitive operation
            if !ModelService.isLoadingModels && !ModelService.isAddingCustomModel {
                updateCustomModelsInUserDefaults()
            } else {
                print("⚠️ Skipping UserDefaults update during sensitive operations")
            }
        } catch {
            print("❌ Error saving models to JSON: \(error)")
        }
    }
    
    /// Load models from models.json file
    func loadModelsFromJSON() {
        // Prevent recursive calls
        guard !ModelService.isLoadingModels else {
            print("⚠️ Avoiding recursive call to loadModelsFromJSON")
            return
        }
        
        // Set the flag
        ModelService.isLoadingModels = true
        defer { ModelService.isLoadingModels = false }
        
        // First make sure the file exists
        ensureModelsJSON()
        
        // During initialization, if the file was just created, don't load or save
        if ModelService.isInitializing && FileManager.default.fileExists(atPath: modelsJSONPath.path) {
            let fileAttributes = try? FileManager.default.attributesOfItem(atPath: modelsJSONPath.path)
            if let creationDate = fileAttributes?[.creationDate] as? Date {
                // If file was created in the last 5 seconds, it's likely fresh
                if Date().timeIntervalSince(creationDate) < 5 {
                    print("📝 models.json was just created, skipping initial load")
                    return
                }
            }
        }
        
        // Verify the file exists
        guard FileManager.default.fileExists(atPath: modelsJSONPath.path) else {
            print("📝 models.json not found, creating empty file")
            saveModelsToJSON()
            return
        }
        
        do {
            // Read the data from the file
            let jsonData = try Data(contentsOf: modelsJSONPath)
            
            // Parse the JSON
            guard let modelDicts = try JSONSerialization.jsonObject(with: jsonData) as? [[String: Any]] else {
                print("❌ Failed to parse models.json: Invalid format")
                return
            }
            
            var addedCount = 0
            
            // Add each model using the addCustomModel method
            for dict in modelDicts {
                guard let id = dict["id"] as? String,
                      let name = dict["name"] as? String,
                      let description = dict["description"] as? String,
                      let size = dict["size"] as? Int,
                      let downloadURL = dict["downloadURL"] as? String else {
                    print("⚠️ Skipping invalid model entry in models.json")
                    continue
                }
                
                // Skip if model already exists
                if availableModels.contains(where: { $0.id == id }) {
                    continue
                }
                
                // Add model using public API
                addCustomModel(
                    name: name,
                    description: description,
                    id: id,
                    size: size,
                    downloadURL: downloadURL
                )
                
                addedCount += 1
                
                // If there's a sourceURL, ensure it's stored in meta.yaml
                if let sourceURL = dict["sourceURL"] as? String, !sourceURL.isEmpty {
                    let modelDir = getModelPath(for: id)
                    ensureConfigurationFile(modelId: id, modelDir: modelDir, sourceURL: sourceURL)
                }
            }
            
            print("✅ Loaded \(addedCount) models from models.json")
        } catch {
            print("❌ Error loading models from JSON: \(error)")
        }
    }
    
    /// Extract source URL from meta.yaml file for a model
    func getSourceURLFromMetaYaml(for modelId: String) -> String? {
        let modelDir = getModelPath(for: modelId)
        let metaYamlPath = modelDir.appendingPathComponent("meta.yaml")
        
        // Check if meta.yaml exists
        guard FileManager.default.fileExists(atPath: metaYamlPath.path) else {
            return nil
        }
        
        do {
            // Read meta.yaml content
            let content = try String(contentsOf: metaYamlPath, encoding: .utf8)
            
            // Look for source_url in the file
            let lines = content.components(separatedBy: .newlines)
            for line in lines {
                if line.contains("source_url:") {
                    // Extract the URL value
                    let parts = line.components(separatedBy: "source_url:")
                    if parts.count >= 2 {
                        let url = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
                        return url
                    }
                }
            }
        } catch {
            print("Error reading meta.yaml for source URL: \(error)")
        }
        
        return nil
    }
    
    /// Add a model to the available models list
    func addModelToAvailable(_ model: Model) {
        // Skip if already exists to prevent unnecessary saves
        if availableModels.contains(where: { $0.id == model.id }) {
            print("⚠️ Model \(model.id) already exists, skipping add")
            return
        }
        
        // Use the addCustomModel function since we can't modify availableModels directly
        addCustomModel(
            name: model.name,
            description: model.description,
            id: model.id,
            size: model.size,
            downloadURL: model.downloadURL
        )
    }
    
    /// Updates custom models in storage
    func updateCustomModelsInUserDefaults() {
        // Prevent recursive calls
        guard !ModelService.isUpdatingUserDefaults else {
            print("⚠️ Avoiding recursive call to updateCustomModelsInUserDefaults")
            return
        }
        
        // Set the flag
        ModelService.isUpdatingUserDefaults = true
        defer { ModelService.isUpdatingUserDefaults = false }
        
        // To be sure initialization is complete
        completeJSONInitialization()
        
        // Only save to models.json, not to UserDefaults
        if !ModelService.isSavingModels {
            // Before saving, update download status based on verification
            updateModelDownloadStatusesBasedOnVerification()
            saveModelsToJSON()
        }
        
        print("📝 Model statuses synced with verification results and saved to models.json")
    }
    
    /// Updates the isDownloaded flag for each model based on actual verification
    private func updateModelDownloadStatusesBasedOnVerification() {
        print("🔄 Syncing model download statuses with verification results")
        
        for (index, model) in availableModels.enumerated() {
            // Check if model is in the process of downloading
            let isDownloading = isModelDownloading(modelId: model.id)
            if isDownloading {
                print("⏳ Skipping verification for model \(model.id) as it's currently downloading")
                continue
            }
            
            // Verify the model using the existing verifyModelFiles method
            let verified = verifyModelFiles(modelId: model.id)
            
            // If download status doesn't match verification, update it
            if model.isDownloaded != verified {
                print("📊 Correcting download status for model \(model.id): \(model.isDownloaded) -> \(verified)")
                
                // Update the model status in the availableModels array
                availableModels[index].isDownloaded = verified
                
                // If the model was previously marked as downloaded but verification failed,
                // we may want to take additional actions (like removing from downloaded list)
                if !verified && model.isDownloaded {
                    print("⚠️ Model \(model.id) was marked as downloaded but verification failed")
                    
                    // Since we can't directly modify downloadedModels, use a helper method
                    // to update the downloaded models list
                    removeFromDownloadedModels(modelId: model.id)
                }
            }
        }
        
        print("✅ Model download statuses synced with verification results")
    }
} 