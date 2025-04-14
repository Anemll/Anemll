// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// ModelManagementViewHelpers.swift

import SwiftUI
import os.signpost

// Extension for direct download features
extension ModelManagementView {
    
    // Helper function to format file sizes in a human-readable format
    func formatFileSize(_ size: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(size))
    }
    
    // Function to download a specific file directly using URLSession
    func downloadFileDirectly(model: Model, filePath: String) async -> Bool {
        // Add signpost for performance monitoring
        let signposter = OSSignposter()
        let signpostID = OSSignpostID(log: .default)
        let state = signposter.beginInterval("DownloadFile", id: signpostID, "Starting \(filePath)")
        
        print("DEBUG: Starting direct download for \(model.id): \(filePath)")
        
        // First, ensure we have meta.yaml to get correct capitalization
        if filePath != "meta.yaml" {
            let metaYamlExists = await ensureMetaYamlExists(for: model)
            if metaYamlExists {
                print("DEBUG: meta.yaml exists, will use for correct capitalization")
            } else {
                print("DEBUG: Could not download meta.yaml, proceeding with original capitalization")
            }
        }
        
        // Move file existence checks to background thread
        let modelDir = modelService.getModelPath(for: model.id)
        let metaYamlPath = modelDir.appendingPathComponent("meta.yaml")
        
        // Try to get model prefix with correct capitalization from meta.yaml
        var modelPrefix: String?
        if filePath != "meta.yaml" {
            // Perform file operations on background thread
            let metaExists = await Task.detached(priority: .background) { @MainActor in
                return FileManager.default.fileExists(atPath: metaYamlPath.path)
            }.value
            
            if metaExists {
                do {
                    // Read file on background thread
                    let metaYamlContent = try await Task.detached(priority: .background) { @MainActor in
                        return try String(contentsOf: metaYamlPath, encoding: .utf8)
                    }.value
                    
                    if let config = try? ModelConfiguration(from: metaYamlContent) {
                        modelPrefix = config.modelPrefix
                        print("DEBUG: Using model prefix with correct capitalization: \(modelPrefix ?? "unknown")")
                    }
                } catch {
                    print("DEBUG: Error reading meta.yaml: \(error.localizedDescription)")
                }
            }
        }
        
        // Extract the case-sensitive repository information from the model's downloadURL
        var originalOwner = "anemll"
        var originalRepo = model.id
        
        // Extract original case owner/repo from the model's downloadURL
        if model.downloadURL.contains("huggingface.co/") || model.downloadURL.contains("huggingface://") {
            let url = model.downloadURL
                .replacingOccurrences(of: "https://huggingface.co/", with: "")
                .replacingOccurrences(of: "huggingface://", with: "")
                .replacingOccurrences(of: "/tree/main", with: "")
                .replacingOccurrences(of: "/blob/main", with: "")
            
            let components = url.components(separatedBy: "/")
            if components.count >= 2 {
                originalOwner = components[0]
                originalRepo = components[1]
                print("DEBUG: Extracted case-sensitive repository: \(originalOwner)/\(originalRepo)")
            }
        }
        
        // Try multiple repository name formats to handle different naming conventions
        // IMPORTANT: First try with the original case-sensitive repository name from downloadURL
        let possibleRepositoryNames = [
            originalRepo,                                 // Original case from downloadURL
            model.id,                                     // Standard format (lowercase)
            model.id.replacingOccurrences(of: "anemll-", with: ""), // Without anemll- prefix
        ]
        
        // Try different branch names
        let possibleBranches = ["main", "master"]
        
        // Try different URL patterns - some repositories might work better with one or the other
        let possibleUrlPatterns = ["resolve", "tree", "raw", "blob"]
        
        // Create destination path
        let destinationPath = modelDir.appendingPathComponent(filePath)
        
        // Create directory structure if it doesn't exist - do this on a background thread
        let directoryPath = destinationPath.deletingLastPathComponent()
        let dirCreated = await Task.detached(priority: .background) {
            do {
                try FileManager.default.createDirectory(
                    at: directoryPath,
                    withIntermediateDirectories: true,
                    attributes: nil
                )
                return true
            } catch {
                print("DEBUG: Failed to create directory structure: \(error.localizedDescription)")
                return false
            }
        }.value
        
        if !dirCreated {
            signposter.endInterval("DownloadFile", state, "Failed to create directory")
            return false
        }
        
        // Create a URLSession configuration with longer timeouts and retry support
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 300
        configuration.waitsForConnectivity = true
        
        let session = URLSession(configuration: configuration)
        
        // If we have the model prefix and this is a model file, use the correct capitalization
        var correctedFilePath = filePath
        if let prefix = modelPrefix, filePath.contains(".mlmodelc") {
            // Try to correct the capitalization using the model prefix
            // Assuming filePath format like: some_prefix_FFN_PF_lut6_chunk_01of08.mlmodelc/weights/weight.bin
            let filePathLower = filePath.lowercased()
            let prefixLower = prefix.lowercased()
            
            if filePathLower.contains(prefixLower) {
                // Replace the lowercase prefix with the correctly capitalized one
                correctedFilePath = filePathLower.replacingOccurrences(of: prefixLower, with: prefix)
                print("DEBUG: Corrected file path capitalization: \(correctedFilePath)")
            }
        }
        
        // Try different URL construction strategies
        for repoName in possibleRepositoryNames {
            for branch in possibleBranches {
                for urlPattern in possibleUrlPatterns {
                    // Strategy 1: Use the original case-sensitive owner from downloadURL
                    let url1 = "https://huggingface.co/\(originalOwner)/\(repoName)/\(urlPattern)/\(branch)/\(correctedFilePath)"
                    
                    // Strategy 2: Fallback to standard anemll prefix if that doesn't work
                    let url2 = "https://huggingface.co/anemll/\(repoName)/\(urlPattern)/\(branch)/\(correctedFilePath)"
                    
                    // Try each URL construction
                    for (index, urlString) in [url1, url2].enumerated() {
                        guard let fileURL = URL(string: urlString) else { continue }
                        
                        print("DEBUG: Trying direct download URL (\(index + 1)/2 pattern:\(urlPattern)): \(urlString)")
                        
                        do {
                            let downloadStart = signposter.beginInterval("URLDownload", id: signpostID, "URL: \(urlString)")
                            let (tempURL, response) = try await session.download(from: fileURL)
                            signposter.endInterval("URLDownload", downloadStart)
                            
                            // Verify it's a valid response
                            guard let httpResponse = response as? HTTPURLResponse else {
                                print("DEBUG: Invalid response type for URL \(urlString)")
                                continue
                            }
                            
                            if httpResponse.statusCode == 404 {
                                print("DEBUG: File not found (404) at \(urlString)")
                                continue // Try next URL pattern
                            }
                            
                            guard httpResponse.statusCode == 200 else {
                                print("DEBUG: HTTP error downloading file. Status: \(httpResponse.statusCode)")
                                continue // Try next URL pattern
                            }
                            
                            // Move the temporary file to its final destination - on background thread
                            let moveSuccess = await Task.detached(priority: .background) {
                                do {
                                    if FileManager.default.fileExists(atPath: destinationPath.path) {
                                        try FileManager.default.removeItem(at: destinationPath)
                                    }
                                    try FileManager.default.moveItem(at: tempURL, to: destinationPath)
                                    return true
                                } catch {
                                    print("DEBUG: Error moving downloaded file: \(error.localizedDescription)")
                                    return false
                                }
                            }.value
                            
                            if !moveSuccess {
                                continue
                            }
                            
                            // Verify the file size
                            let fileSize = await Task.detached(priority: .background) {
                                do {
                                    let attributes = try FileManager.default.attributesOfItem(atPath: destinationPath.path)
                                    return attributes[FileAttributeKey.size] as? UInt64
                                } catch {
                                    print("DEBUG: Error getting file size: \(error.localizedDescription)")
                                    return nil
                                }
                            }.value
                            
                            if let fileSize = fileSize {
                                print("DEBUG: Successfully downloaded file directly: \(filePath) - size: \(formatFileSize(Int(fileSize)))")
                                if fileSize < 1024 && filePath.hasSuffix("weight.bin") {
                                    print("DEBUG: WARNING: Downloaded weight file is suspiciously small: \(fileSize) bytes")
                                    signposter.endInterval("DownloadFile", state, "File too small")
                                    return false
                                }
                                
                                // Success - found and downloaded a valid file
                                signposter.endInterval("DownloadFile", state, "Success")
                                return true
                            }
                        } catch {
                            print("DEBUG: Error downloading file from \(urlString): \(error.localizedDescription)")
                            // Continue to try the next URL pattern
                        }
                    }
                }
            }
        }
        
        // If we get here, all download attempts failed
        print("DEBUG: All download attempts failed for \(filePath)")
        signposter.endInterval("DownloadFile", state, "All attempts failed")
        return false
    }
    
    // Function to ensure that the meta.yaml file exists
    func ensureMetaYamlExists(for model: Model) async -> Bool {
        let modelDir = modelService.getModelPath(for: model.id)
        let metaPath = modelDir.appendingPathComponent("meta.yaml")
        
        // Check if meta.yaml exists
        if FileManager.default.fileExists(atPath: metaPath.path) {
            print("DEBUG: meta.yaml exists for model \(model.id)")
            return true
        }
        
        print("DEBUG: meta.yaml missing for model \(model.id), attempting to download it")
        
        // Try to download meta.yaml
        return await downloadFileDirectly(model: model, filePath: "meta.yaml")
    }
    
    // Function to generate a diagnostic report for debugging
    // This should replace the existing implementation if it exists
    func generateModelDiagnosticReport(for model: Model) -> String {
        let modelDir = modelService.getModelPath(for: model.id)
        var report = """
        === MODEL DIAGNOSTIC REPORT ===
        Model ID: \(model.id)
        Model Name: \(model.name)
        Downloaded: \(model.isDownloaded)
        Has Placeholders: \(model.hasPlaceholders)
        
        Model Path: \(modelDir.path)
        
        """
        
        // Check if model directory exists
        if FileManager.default.fileExists(atPath: modelDir.path) {
            report += "Model directory exists: Yes\n"
            
            // Check meta.yaml
            let metaPath = modelDir.appendingPathComponent("meta.yaml")
            if FileManager.default.fileExists(atPath: metaPath.path) {
                report += "meta.yaml exists: Yes\n"
                
                // Get meta.yaml contents
                do {
                    let metaContents = try String(contentsOf: metaPath, encoding: .utf8)
                    report += "meta.yaml contents:\n\(metaContents)\n"
                    
                    // Try to parse required files from meta.yaml
                    var requiredFiles: [String] = []
                    
                    // Check for top-level files section
                    var filesSection = false
                    var inModelInfoSection = false
                    var inNestedFilesSection = false
                    
                    for line in metaContents.components(separatedBy: CharacterSet.newlines) {
                        let trimmedLine = line.trimmingCharacters(in: CharacterSet.whitespaces)
                        
                        // Track when we're in the model_info section
                        if trimmedLine == "model_info:" {
                            inModelInfoSection = true
                            continue
                        }
                        
                        // Check for files: section (top level or within model_info)
                        if trimmedLine == "files:" {
                            filesSection = true
                            continue
                        }
                        
                        // Check for nested files section within model_info
                        if inModelInfoSection && trimmedLine == "  files:" {
                            inNestedFilesSection = true
                            continue
                        }
                        
                        // Capture files in standard files section
                        if filesSection && (trimmedLine.hasPrefix("- ") || trimmedLine.hasPrefix("  - ")) {
                            let file = trimmedLine.trimmingCharacters(in: CharacterSet.whitespaces).replacingOccurrences(of: "- ", with: "")
                            requiredFiles.append(file)
                        }
                        
                        // Capture files in nested files section within model_info
                        if inNestedFilesSection && (trimmedLine.hasPrefix("  - ") || trimmedLine.hasPrefix("    - ")) {
                            let file = trimmedLine.trimmingCharacters(in: CharacterSet.whitespaces).replacingOccurrences(of: "- ", with: "")
                            requiredFiles.append(file)
                        }
                        
                        // Exit model_info section when a new top-level section starts
                        if inModelInfoSection && !trimmedLine.isEmpty && !trimmedLine.hasPrefix(" ") && !trimmedLine.hasPrefix("\t") && trimmedLine != "model_info:" {
                            inModelInfoSection = false
                        }
                        
                        // Exit files section when we hit a non-indented, non-empty line
                        if filesSection && !trimmedLine.isEmpty && !trimmedLine.hasPrefix(" ") && !trimmedLine.hasPrefix("\t") && !trimmedLine.hasPrefix("-") {
                            filesSection = false
                        }
                        
                        // Exit nested files section when indentation changes
                        if inNestedFilesSection && !trimmedLine.isEmpty && !trimmedLine.hasPrefix("    ") && !trimmedLine.hasPrefix("  ") {
                            inNestedFilesSection = false
                        }
                    }
                    
                    // Also try to parse the files based on the model configuration (for CoreML models)
                    do {
                        let modelConfig = try ModelConfiguration(from: metaContents, modelPath: modelDir.path)
                        let modelService = ModelService.shared
                        let inferredFiles = modelService.getRequiredFiles(from: modelConfig)
                        
                        // Add any files that aren't already in our list
                        for file in inferredFiles {
                            if !requiredFiles.contains(file) {
                                requiredFiles.append(file)
                            }
                        }
                    } catch {
                        report += "Error inferring required files from configuration: \(error.localizedDescription)\n"
                    }
                    
                    report += "\nRequired files from meta.yaml (\(requiredFiles.count)):\n"
                    if !requiredFiles.isEmpty {
                        for file in requiredFiles {
                            report += "- \(file)\n"
                        }
                    }
                    
                    // Check each required file
                    for file in requiredFiles {
                        let filePath = modelDir.appendingPathComponent(file)
                        let exists = FileManager.default.fileExists(atPath: filePath.path)
                        
                        var fileInfo = "- \(file): \(exists ? "Exists" : "Missing")"
                        
                        if exists {
                            do {
                                let attrs = try FileManager.default.attributesOfItem(atPath: filePath.path)
                                if let size = attrs[FileAttributeKey.size] as? UInt64 {
                                    fileInfo += ", Size: \(formatFileSize(Int(size)))"
                                    
                                    if file.hasSuffix("weight.bin") && size < 1_048_576 {
                                        fileInfo += " (SUSPICIOUS - TOO SMALL!)"
                                    }
                                } else {
                                    fileInfo += ", Size: Unknown"
                                }
                            } catch {
                                fileInfo += ", Error getting size: \(error.localizedDescription)"
                            }
                        }
                        
                        report += "\(fileInfo)\n"
                    }
                } catch {
                    report += "Error reading meta.yaml: \(error.localizedDescription)\n"
                }
            } else {
                report += "meta.yaml exists: No\n"
            }
            
            // Report free disk space
            do {
                let fileSystemAttrs = try FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory())
                if let freeSpace = fileSystemAttrs[FileAttributeKey.systemFreeSize] as? NSNumber {
                    report += "\nAvailable disk space: \(formatFileSize(freeSpace.intValue))\n"
                }
            } catch {
                report += "\nError getting disk space: \(error.localizedDescription)\n"
            }
        } else {
            report += "Model directory exists: No\n"
        }
        
        report += "\n=== END REPORT ===\n"
        return report
    }
    
    // Function to verify downloaded files with better error handling
    func verifyDownloadedFiles(model: Model, expectedFiles: [String]) async -> (allFilesPresent: Bool, problemFiles: [String]) {
        let modelDir = modelService.getModelPath(for: model.id)
        var problemFiles: [String] = []
        
        print("DEBUG: Verifying downloaded files for \(model.id)...")
        
        for file in expectedFiles {
            let filePath = modelDir.appendingPathComponent(file)
            let fileExists = FileManager.default.fileExists(atPath: filePath.path)
            
            if !fileExists {
                print("DEBUG: Missing file after download: \(file)")
                problemFiles.append(file)
                continue
            }
            
            // For weight files, also check size
            if file.hasSuffix("weight.bin") {
                do {
                    let attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
                    if let fileSize = attributes[FileAttributeKey.size] as? UInt64 {
                        // Log file size
                        print("DEBUG: File size for \(file): \(formatFileSize(Int(fileSize)))")
                        
                        // Check if file is suspiciously small (less than 1MB for weight files)
                        if fileSize < 1_048_576 { // 1MB
                            print("DEBUG: Suspiciously small weight file: \(file) - size: \(formatFileSize(Int(fileSize)))")
                            problemFiles.append(file)
                        }
                    }
                } catch {
                    print("DEBUG: Error checking file size for \(file): \(error.localizedDescription)")
                    problemFiles.append(file)
                }
            }
        }
        
        return (problemFiles.isEmpty, problemFiles)
    }
} 
