// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// Model.swift

import Foundation
import SwiftUI

/// Model class representing a language model
class Model: Identifiable, ObservableObject {
    let id: String
    let name: String
    let description: String
    @Published var size: Int
    @Published var downloadURL: String
    @Published var isDownloaded: Bool = false
    
    // Flag to indicate if the model has placeholder files
    @Published public var hasPlaceholders: Bool = false
    
    // These properties need to be @Published to update the UI
    @Published var customStatus: String?
    
    init(id: String, name: String, description: String, size: Int, downloadURL: String) {
        self.id = id
        self.name = name
        self.description = description
        self.size = size
        self.downloadURL = downloadURL
    }
    
    /// Checks if the model is downloaded
    func checkIsDownloaded() -> Bool {
        // Get the model path directly without using ModelService.shared
        // This avoids recursive initialization issues
        let fileManager = FileManager.default
        let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelsDirectory = documentsDirectory.appendingPathComponent("Models")
        
        let modelPath: URL
        if id == "llama-3.2-1b" {
            // Special case for default model
            modelPath = modelsDirectory.appendingPathComponent("llama_3_2_1b_iosv2_0")
        } else {
            // For other models, use the sanitized ID
            let sanitizedId = id.replacingOccurrences(of: "/", with: "_")
                .replacingOccurrences(of: "\\", with: "_")
                .replacingOccurrences(of: ":", with: "_")
                .replacingOccurrences(of: "*", with: "_")
                .replacingOccurrences(of: "?", with: "_")
                .replacingOccurrences(of: "\"", with: "_")
                .replacingOccurrences(of: "<", with: "_")
                .replacingOccurrences(of: ">", with: "_")
                .replacingOccurrences(of: "|", with: "_")
            modelPath = modelsDirectory.appendingPathComponent(sanitizedId)
        }
        
        // First check if directory exists at all
        let exists = fileManager.fileExists(atPath: modelPath.path)
        
        if !exists {
            print("📂 Model directory does not exist at \(modelPath.path)")
            return false
        }
        
        // Check that directory has actual content
        do {
            let contents = try fileManager.contentsOfDirectory(atPath: modelPath.path)
            
            // If directory is completely empty, consider model not downloaded
            if contents.isEmpty {
                print("📂 Model directory exists but is empty at \(modelPath.path)")
                return false
            }
            
            // Check if directory has minimal required structure
            let hasMLModelC = contents.contains { $0.hasSuffix(".mlmodelc") }
            let hasTokenizer = contents.contains { $0 == "tokenizer.json" }
            let hasConfig = contents.contains { $0 == "config.json" }
            
            // Only mark as downloaded if at least one of these exists
            let hasMinimalContent = hasMLModelC || hasTokenizer || hasConfig
            
            if !hasMinimalContent {
                print("📂 Model directory exists but missing essential files at \(modelPath.path)")
                return false
            }
            
            // Directory exists and has minimum content
            return true
        } catch {
            print("📂 Error checking model directory contents: \(error)")
            // If we can't check contents, fall back to directory existence
            return exists
        }
    }
    
    /// Refreshes the download status of the model
    func refreshDownloadStatus() {
        // Use Task.detached to move the file system check off the main thread
        Task.detached {
            let wasDownloaded = self.isDownloaded
            let exists = self.checkIsDownloaded()
            
            // Update state on the MainActor
            await MainActor.run {
                self.isDownloaded = exists
                
                if wasDownloaded != exists {
                    print("📊 Model \(self.id) download status changed: \(wasDownloaded) -> \(exists)")
                }
            }
        }
    }
    
    /// Gets a formatted string for the model size
    func getFormattedSize() -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(size))
    }
    
    // Helper method to get the status display string
    func getStatusDisplayString() -> String {
        if let customStatus = customStatus {
            return customStatus
        } else if isDownloaded {
            return "Downloaded"
        } else {
            return "Not Downloaded"
        }
    }
    
    // Helper method to check if the model can be used for inference
    func canUseForInference() -> Bool {
        return isDownloaded && !hasPlaceholders
    }
}