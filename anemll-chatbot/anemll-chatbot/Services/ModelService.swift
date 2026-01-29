// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// ModelService.swift

import Foundation
import Combine
import CoreML
import Compression
import Yams
#if os(macOS)
import AppKit
#else
import UIKit
#endif
// import ZIPFoundation - not linked to target, using native unzip instead

// Add ModelError enum at the top level
enum ModelError: Error {
    case downloadFailed(String)
    case invalidModelFormat(String)
    case missingRequiredFiles(String)
}

// Using @unchecked Sendable since we're manually managing thread safety
final class ModelService: NSObject, URLSessionDownloadDelegate, ObservableObject, @unchecked Sendable {
    // Use a more robust singleton pattern with dispatch_once semantics
    private static let _shared = ModelService()
    
    // Public accessor that ensures initialization is complete before returning
    static var shared: ModelService {
        // Print only once when first accessed
        struct Static {
            static var token: Int = 0
            static var isInitializing = false
        }
        
        if Static.token == 0 {
            DispatchQueue.once(token: &Static.token) {
                print("🚀 First access to ModelService.shared")
                Static.isInitializing = true
                // The initialization happens in init() which is called when _shared is created
                Static.isInitializing = false
            }
        }
        
        return _shared
    }
    
    // File manager and storage properties
    private let fileManager = FileManager.default
    private var sharedDirectory: URL
    internal var modelStorageDirectory: URL
    
    // Keys for UserDefaults
    private let selectedModelIdKey = "selectedModelId"
    
    // Published properties
    // Note: We're using @unchecked Sendable and manually ensuring thread safety
    // for these properties by only modifying them on the main thread
    @Published private(set) var selectedModel: Model?
    @Published private(set) var availableModels: [Model] = []
    @Published private(set) var downloadedModels: [Model] = []
    
    // Device type enum for better platform handling
    enum DeviceType {
        case mac
        case macCatalyst
        case iPad
        case iPhone
        case other
        
        static var current: DeviceType {
            #if os(macOS)
            return .mac
            #elseif targetEnvironment(macCatalyst)
            return .macCatalyst
            #else
            let device = UIDevice.current
            if device.userInterfaceIdiom == .pad {
                return .iPad
            } else if device.userInterfaceIdiom == .phone {
                return .iPhone
            } else {
                return .other
            }
            #endif
        }
    }
    
    // Hugging Face model repository information
    private struct HuggingFaceRepo {
        // Original case-sensitive components (for URL construction)
        let owner: String
        let repo: String
        let branch: String
        
        // Lowercase versions for case-insensitive comparisons
        let ownerLowercase: String
        let repoLowercase: String
        
        init(owner: String, repo: String, branch: String) {
            self.owner = owner  // Preserve original case
            self.repo = repo    // Preserve original case
            self.branch = branch
            self.ownerLowercase = owner.lowercased()
            self.repoLowercase = repo.lowercased()
        }
        
        var apiUrl: String {
            return "https://huggingface.co/api/models/\(owner)/\(repo)"
        }
        
        var filesApiUrl: String {
            return "https://huggingface.co/api/models/\(owner)/\(repo)/tree/\(branch)"
        }
        
        var downloadBaseUrl: String {
            return "https://huggingface.co/\(owner)/\(repo)/resolve/\(branch)/"
        }
    }
    
    private let defaultHuggingFaceRepo = HuggingFaceRepo(
        owner: "anemll",
        repo: "anemll-llama-3.2-1B-iOSv2.0",
        branch: "main"
    )
    
    private var downloadTasks: [String: Any] = [:]  // Changed from URLSessionDownloadTask to Any to support CombinedDownloadTask
    private var progressObservers: [String: ((Double) -> Void)] = [:]
    private var fileProgressObservers: [String: ((String, Double) -> Void)] = [:]
    private var completionHandlers: [String: ((Bool) -> Void)] = [:]
    private var cancellables = Set<AnyCancellable>()
    private let deviceType = DeviceType.current
    private var currentDownloadingFiles: [String: String] = [:]
    private var downloadProgress: [String: Double] = [:]
    private var downloadActivityTimers: [String: Timer] = [:] // For pulsing animation during active downloads
    private var isDownloadActive: [String: Bool] = [:]
    private var downloadProgressObservers: [String: [NSKeyValueObservation]] = [:] // For progress observers

    // Progress smoothing: track max values to prevent UI jumping backward
    private var maxDownloadedBytes: [String: Int64] = [:]
    private var maxDownloadProgress: [String: Double] = [:]

    // MARK: - Enhanced Download Tracking (for proper cancel and resume support)

    /// Tracks individual file download tasks by a unique key (modelId:fileName)
    private var fileDownloadTasks: [String: URLSessionDownloadTask] = [:]

    /// Stores resume data for interrupted downloads (key: modelId:fileName)
    private var resumeDataStore: [String: Data] = [:]

    /// Tracks expected file sizes from Content-Length headers (for integrity validation)
    private var expectedFileSizes: [String: Int64] = [:]

    /// Tracks bytes downloaded per file for accurate progress (key: modelId:fileName)
    private var bytesDownloadedPerFile: [String: Int64] = [:]

    /// Tracks total bytes expected for a model download
    private var totalBytesExpectedForModel: [String: Int64] = [:]

    /// Tracks total bytes downloaded for a model
    private var totalBytesDownloadedForModel: [String: Int64] = [:]

    /// Error messages for display in UI
    private var downloadErrorMessages: [String: String] = [:]

    /// Tracks download start time for ETA calculation (key: modelId)
    private var downloadStartTimes: [String: Date] = [:]

    /// Tracks last progress update time for stall detection (key: modelId)
    private var lastProgressUpdateTime: [String: Date] = [:]

    /// Watchdog timers for detecting stalled downloads (key: modelId)
    private var downloadWatchdogTimers: [String: Timer] = [:]

    /// Retry count per file (key: modelId:fileName)
    private var fileRetryCount: [String: Int] = [:]

    /// Maximum retry attempts for failed downloads
    private let maxRetryAttempts = 3

    /// Active dispatch groups for downloads (key: modelId) - kept alive to prevent semaphore crash
    private var activeDownloadGroups: [String: DispatchGroup] = [:]

    /// Active semaphores for downloads (key: modelId) - kept alive to prevent semaphore crash
    private var activeDownloadSemaphores: [String: DispatchSemaphore] = [:]

    /// Active queues for downloads (key: modelId) - kept alive to prevent premature deallocation
    private var activeDownloadQueues: [String: DispatchQueue] = [:]

    /// Caches model display names to avoid repeated lookups (key: modelId)
    private var modelDisplayNames: [String: String] = [:]

    /// Shared URLSession for file downloads with proper delegate handling
    private lazy var fileDownloadSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60      // 1 minute for initial connection
        config.timeoutIntervalForResource = 3600   // 1 hour for large files
        config.waitsForConnectivity = true
        config.allowsCellularAccess = true
        // Enable HTTP pipelining for better performance
        config.httpMaximumConnectionsPerHost = 4
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    /// Recursive lock for thread-safe access to download tracking dictionaries (allows nested calls from same thread)
    private let downloadTrackingLock = NSRecursiveLock()

    /// Thread-safe access helper - use this for all access to tracking dictionaries
    private func withDownloadLock<T>(_ block: () -> T) -> T {
        downloadTrackingLock.lock()
        defer { downloadTrackingLock.unlock() }
        return block()
    }

    /// Mapping from URLSessionTask to file info for delegate callbacks
    private var taskToFileInfo: [Int: (modelId: String, fileName: String, destination: URL)] = [:]

    /// Completion handlers for individual file downloads
    private var fileCompletionHandlers: [String: (Bool) -> Void] = [:]
    
    // Model download status enum
    enum ModelDownloadStatus {
        case notDownloaded
        case downloading
        case downloaded
        case partiallyDownloaded
        case failed
    }
    
    // Track download status for each model
    private var modelDownloadStatus: [String: ModelDownloadStatus] = [:]
    
    // Add this property near the other private properties in ModelService class
    private var activeDownloads: Int = 0 {
        didSet {
            updateIdleTimer()
        }
    }
    
    override init() {
        print("📱 Starting ModelService initialization...")
        
        // Set up platform-specific storage location
        #if os(macOS) || targetEnvironment(macCatalyst)
        // Use /Documents/Models for both macOS and Mac Catalyst
        let homeDirectory: URL
        #if os(macOS)
        homeDirectory = fileManager.homeDirectoryForCurrentUser
        #else
        // For Mac Catalyst, we need to get the home directory differently
        if let homeDir = ProcessInfo.processInfo.environment["HOME"] {
            homeDirectory = URL(fileURLWithPath: homeDir)
        } else {
            // Fallback to user domain directory
            homeDirectory = fileManager.urls(for: .userDirectory, in: .userDomainMask).first!
        }
        #endif
        
        let cacheDirectory = homeDirectory.appendingPathComponent("Documents")
        sharedDirectory = cacheDirectory
        modelStorageDirectory = cacheDirectory.appendingPathComponent("")
        
        // Create directory if it doesn't exist
        if !fileManager.fileExists(atPath: cacheDirectory.path) {
            do {
                try fileManager.createDirectory(at: cacheDirectory, withIntermediateDirectories: true, attributes: nil)
                print("Created .cache directory at \(cacheDirectory.path)")
            } catch {
                print("Error creating .cache directory: \(error)")
                // If we can't create the .cache directory, we might need to request permission
                // For now, we'll log the error and continue with a fallback
            }
        }
        
        if !fileManager.fileExists(atPath: modelStorageDirectory.path) {
            do {
                try fileManager.createDirectory(at: modelStorageDirectory, withIntermediateDirectories: true, attributes: nil)
                print("Created anemll directory at \(modelStorageDirectory.path)")
            } catch {
                print("Error creating anemll directory: \(error)")
                // If we can't create the directory, we might need to request permission
                // For now, we'll log the error and continue with a fallback
            }
        }
        #else
        // Use document directory for iOS/iPadOS (same as ChatService)
        let documentDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        sharedDirectory = documentDirectory
        modelStorageDirectory = documentDirectory.appendingPathComponent("Models")
        
        // Create directory if it doesn't exist
        if !fileManager.fileExists(atPath: modelStorageDirectory.path) {
            do {
                try fileManager.createDirectory(at: modelStorageDirectory, withIntermediateDirectories: true, attributes: nil)
                print("Created Models directory at \(modelStorageDirectory.path)")
            } catch {
                print("Error creating Models directory: \(error)")
            }
        }
        
        // Exclude from iCloud backup
        var resourceValues = URLResourceValues()
        resourceValues.isExcludedFromBackup = true
        do {
            var directoryURL = modelStorageDirectory
            try directoryURL.setResourceValues(resourceValues)
            print("Excluded Models directory from iCloud backup")
        } catch {
            print("Error excluding Models directory from iCloud backup: \(error)")
        }
        #endif
        
        // Use the recommended URLSession initialization instead of the deprecated init()
        super.init()
        
        print("📂 Model storage directory set up at: \(modelStorageDirectory.path)")
        
        // Initialize with empty arrays first
        self.availableModels = []
        self.downloadedModels = []
        
        // Complete initialization in a safer way
        setupModelService()
    }
    
    private func setupModelService() {
        // Use a static flag to ensure this is only called once
        struct Static {
            static var isSetup = false
        }
        
        // Thread-safe check and set
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }
        
        if Static.isSetup {
            print("⚠️ setupModelService already called, skipping")
            return
        }
        
        print("🔄 Setting up ModelService...")
        
        // Clear any custom models from UserDefaults as we're no longer using it
        UserDefaults.standard.removeObject(forKey: "customModels")
        print("🧹 Cleared custom models from UserDefaults")
        
        // Add default models
        setupDefaultModels()
        
        // Clean up any duplicate models
        cleanupDuplicateModels()
        
        // IMPORTANT: Don't call model.checkIsDownloaded() during initialization
        // Instead, directly check if the model directories exist
        for model in availableModels {
            // Check if model directory exists directly
            let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
            let modelsDirectory = documentsDirectory.appendingPathComponent("Models")
            
            let modelPath: URL
            if model.id == "llama-3.2-1b" {
                // Special case for default model
                modelPath = modelsDirectory.appendingPathComponent("llama_3_2_1b_iosv2_0")
            } else {
                // For other models, use the sanitized ID
                let sanitizedId = sanitizeModelId(model.id)
                modelPath = modelsDirectory.appendingPathComponent(sanitizedId)
            }
            
            model.isDownloaded = fileManager.fileExists(atPath: modelPath.path)
            print("Model \(model.id) download status set to: \(model.isDownloaded)")
        }
        
        // Initialize downloadedModels
        downloadedModels = availableModels.filter { $0.isDownloaded }
        
        // Ensure models.json exists and is properly set up before loading custom models
        print("Ensuring models.json is set up...")
        ensureModelsJSON()
        
        // Load custom models
        loadCustomModels()
        
        // Load the previously selected model
        loadSelectedModel()
        
        print("Device type: \(deviceType)")
        print("Models storage location: \(modelStorageDirectory.path)")
        
        // Check if we have write access to the directory
        let testFilePath = modelStorageDirectory.appendingPathComponent("write_test.txt")
        do {
            try "Test write access".write(to: testFilePath, atomically: true, encoding: .utf8)
            try fileManager.removeItem(at: testFilePath)
            print("Write access to models directory confirmed")
        } catch {
            print("WARNING: No write access to models directory: \(error)")
            // Here we could implement a permission request or fallback strategy
        }
        
        // Don't fetch model information from Hugging Face on startup
        // This should only happen when explicitly requested
        // fetchModelInformation()
        
        Static.isSetup = true
        print("✅ ModelService setup complete")
        
        // Mark JSON initialization as complete to allow normal operations
        completeJSONInitialization()
        
        // Preload the selected model immediately
        preloadSelectedModel()
    }
    
    // Add a new method to explicitly fetch model information when needed
    public func fetchModelInformationIfNeeded() {
        print("🌐 Explicitly fetching model information from Hugging Face")
        fetchModelInformation()
    }
    
    /// Preloads the selected model to make it ready for inference
    public func preloadSelectedModel() {
        guard let selectedModel = selectedModel else {
            print("⚠️ No selected model available for preloading")
            return
        }
        
        print("🔄 Attempting to preload selected model: \(selectedModel.name)")
        
        // First, check if the model directory exists at the expected path
        let modelPath = getModelPath(for: selectedModel.id)
        let directoryExists = fileManager.fileExists(atPath: modelPath.path)
        
        if !directoryExists {
            print("⚠️ Model directory does not exist at \(modelPath.path)")
            selectedModel.isDownloaded = false
            return
        }
        
        // Update the model's download status based on the directory existence
        selectedModel.isDownloaded = true
        
        // Verify the model files
        let isValid = verifyModelFiles(modelId: selectedModel.id)
        if !isValid {
            print("⚠️ Selected model verification failed, cannot preload")
            return
        }
        
        print("📂 Model path for preloading: \(modelPath.path)")
        
        // Start the loading process with high priority
        Task(priority: .userInitiated) {
            do {
                print("🚀 Starting model preloading for: \(selectedModel.id)")
                
                // Set loading state in InferenceService
                await MainActor.run {
                    InferenceService.shared.loadingStatus = "Starting model loading..."
                    InferenceService.shared.isLoadingModel = true
                    InferenceService.shared.loadingProgress = 0.05
                    
                    // Post notification that model loading has started
                    NotificationCenter.default.post(
                        name: Notification.Name("ModelLoadingStarted"),
                        object: selectedModel.id
                    )
                }
                
                // Start loading the model
                try await InferenceService.shared.loadModel(modelId: selectedModel.id, from: modelPath)
                
                print("✅ Model preloaded successfully: \(selectedModel.id)")
            } catch {
                print("❌ Error preloading model: \(error.localizedDescription)")
                
                // Don't post failure notifications for cancellation errors
                if error is CancellationError {
                    print("🔄 Model preloading was cancelled - not posting error notification")
                    
                    // Just reset loading state without posting error
                    await MainActor.run {
                        InferenceService.shared.isLoadingModel = false
                        InferenceService.shared.loadingProgress = 0
                        InferenceService.shared.loadingStatus = ""
                    }
                    return
                }
                
                // Reset loading state
                await MainActor.run {
                    InferenceService.shared.isLoadingModel = false
                    InferenceService.shared.loadingProgress = 0
                    InferenceService.shared.loadingStatus = "Error: \(error.localizedDescription)"
                    
                    // Post notification that model loading failed (only for non-cancellation errors)
                    NotificationCenter.default.post(
                        name: Notification.Name("ModelLoadingFailed"),
                        object: selectedModel.id,
                        userInfo: ["error": error.localizedDescription]
                    )
                }
            }
        }
    }
    
    // Track which models have interrupted loading
    private var interruptedLoadings = Set<String>()
    
    /// Interrupts the loading of a model
    public func interruptModelLoading(modelId: String) {
        print("⚠️ Interrupting model loading for: \(modelId)")
        interruptedLoadings.insert(modelId)
        
        // You might need additional logic here to actually stop the loading process
        // depending on your ML framework
    }
    
    /// Checks if loading was interrupted for a specific model
    private func isLoadingInterrupted(for modelId: String) -> Bool {
        return interruptedLoadings.contains(modelId)
    }
    
    /// Resets the interrupted state for a model
    public func resetInterruptedState(for modelId: String) {
        interruptedLoadings.remove(modelId)
    }
    
    private func setupDefaultModels() {
        print("📚 Setting up default models...")
        
        // Add default model (Llama 3.2 1B)
        let defaultModel = Model(
            id: "llama-3.2-1b",
            name: "Llama 3.2 1B",
            description: "Llama 3.2 1B model optimized for iOS/macOS",
            size: 1_600_000_000, // Approximate size in bytes (1.6GB), will be updated once download completes
            downloadURL: "https://huggingface.co/anemll/anemll-llama-3.2-1B-iOSv2.0"
        )
        
        // Check if model is already downloaded using the consistent path
        // IMPORTANT: Don't use getModelPath here as it might cause recursive initialization
        let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelsDirectory = documentsDirectory.appendingPathComponent("Models")
        let defaultModelPath = modelsDirectory.appendingPathComponent("llama_3_2_1b_iosv2_0")
        
        defaultModel.isDownloaded = fileManager.fileExists(atPath: defaultModelPath.path)
        
        // If the model is downloaded, try to get the actual size from disk
        if defaultModel.isDownloaded {
            do {
                // Get the size of the directory by summing up all files
                let contents = try fileManager.contentsOfDirectory(at: defaultModelPath, includingPropertiesForKeys: [.fileSizeKey])
                var totalSize: Int = 0
                
                for fileURL in contents {
                    let attributes = try fileManager.attributesOfItem(atPath: fileURL.path)
                    if let size = attributes[.size] as? NSNumber {
                        totalSize += size.intValue
                    }
                }
                
                if totalSize > 0 {
                    defaultModel.size = totalSize
                    print("Updated default model size from disk: \(formatFileSize(defaultModel.size))")
                }
            } catch {
                print("Could not get size of downloaded model: \(error)")
            }
        }
        
        // Add to available models
        availableModels.append(defaultModel)
        
        print("Added default model: \(defaultModel.name), isDownloaded: \(defaultModel.isDownloaded)")
    }
    
    // MARK: - Hugging Face API Integration
    
    private func fetchModelInformation() {
        guard let url = URL(string: defaultHuggingFaceRepo.apiUrl) else {
            print("Invalid Hugging Face API URL")
            return
        }
        
        print("Fetching model information from: \(url.absoluteString)")
        
        URLSession.shared.dataTask(with: url) { [weak self] data, response, error in
            guard let self = self else { return }
            
            if let error = error {
                print("Error fetching model information: \(error.localizedDescription)")
                return
            }
            
            if let httpResponse = response as? HTTPURLResponse {
                print("Model API response status: \(httpResponse.statusCode)")
                
                if httpResponse.statusCode != 200 {
                    print("Error: Unexpected status code \(httpResponse.statusCode)")
                    if let data = data, let responseString = String(data: data, encoding: .utf8) {
                        print("Response body: \(responseString)")
                    }
                    return
                }
            }
            
            guard let data = data else {
                print("Error: No data received from model API")
                return
            }
            
            print("Received \(data.count) bytes of model data")
            
            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    print("Successfully parsed model information")
                    self.updateModelFromHuggingFace(json)
                } else {
                    print("Error: Unexpected JSON format")
                    if let responseString = String(data: data, encoding: .utf8) {
                        print("Response body: \(responseString)")
                    }
                }
            } catch {
                print("Error parsing model information: \(error.localizedDescription)")
                if let responseString = String(data: data, encoding: .utf8) {
                    print("Response body: \(responseString)")
                }
            }
        }.resume()
        
        // Also fetch the file list to determine what files we need to download
        fetchModelFiles()
    }
    
    private func fetchModelFiles() {
        guard let url = URL(string: defaultHuggingFaceRepo.filesApiUrl) else {
            print("Invalid Hugging Face Files API URL")
            return
        }
        
        print("Fetching files from: \(url.absoluteString)")
        
        URLSession.shared.dataTask(with: url) { [weak self] data, response, error in
            guard let self = self else { return }
            
            if let error = error {
                print("Error fetching model files: \(error.localizedDescription)")
                return
            }
            
            if let httpResponse = response as? HTTPURLResponse {
                print("File API response status: \(httpResponse.statusCode)")
                
                if httpResponse.statusCode != 200 {
                    print("Error: Unexpected status code \(httpResponse.statusCode)")
                    if let data = data, let responseString = String(data: data, encoding: .utf8) {
                        print("Response body: \(responseString)")
                    }
                    return
                }
            }
            
            guard let data = data else {
                print("Error: No data received from file API")
                return
            }
            
            print("Received \(data.count) bytes of file data")
            
            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                    print("Successfully parsed file list with \(json.count) entries")
                    self.processModelFiles(json)
                } else {
                    print("Error: Unexpected JSON format")
                    if let responseString = String(data: data, encoding: .utf8) {
                        print("Response body: \(responseString)")
                    }
                }
            } catch {
                print("Error parsing model files: \(error.localizedDescription)")
                if let responseString = String(data: data, encoding: .utf8) {
                    print("Response body: \(responseString)")
                }
            }
        }.resume()
    }
    
    private func updateModelFromHuggingFace(_ modelInfo: [String: Any]) {
        if availableModels.first(where: { $0.id == "llama-3.2-1b" }) == nil {
            return
        }
        
        DispatchQueue.main.async(execute: { [weak self] in
            guard self != nil else { return }
            
            if let downloads = modelInfo["downloads"] as? Int {
                print("Model downloads: \(downloads)")
            }
            
            if let lastModified = modelInfo["lastModified"] as? String {
                print("Last modified: \(lastModified)")
            }
            
            // Update model information if needed
            if let tags = modelInfo["tags"] as? [String], !tags.isEmpty {
                print("Model tags: \(tags.joined(separator: ", "))")
            }
        })
    }
    
    private func processModelFiles(_ files: [[String: Any]]) {
        // Filter for required model files (meta.yaml and CoreML model files)
        var requiredFiles: [String] = []
        var totalSize: Int64 = 0
        
        print("Found \(files.count) files in the repository")
        
        // First, look for specific file types we need
        for file in files {
            if let path = file["path"] as? String {
                print("Found file: \(path)")
                
                // Check for various file types we might need
                if path.hasSuffix(".mlmodelc") || 
                   path.hasSuffix(".mlmodel") || 
                   path.hasSuffix(".mlpackage") || 
                   path == "meta.yaml" || 
                   path.hasSuffix(".yaml") || 
                   path.hasSuffix(".json") || 
                   path.hasSuffix(".bin") || 
                   path.hasSuffix(".zip") {
                    
                    requiredFiles.append(path)
                    if let size = file["size"] as? Int64 {
                        totalSize += size
                    }
                }
            }
        }
        
        // If we didn't find any specific files, just download everything
        if requiredFiles.isEmpty {
            print("No specific model files found, downloading all files")
            for file in files {
                if let path = file["path"] as? String,
                   !path.hasSuffix("/") { // Skip directories
                    requiredFiles.append(path)
                    if let size = file["size"] as? Int64 {
                        totalSize += size
                    }
                }
            }
        }
        
        print("Required files: \(requiredFiles.joined(separator: ", "))")
        print("Total download size: \(formatFileSize(Int(totalSize)))")
        
        // Update model size and download URL
        if let model = availableModels.first(where: { $0.id == "llama-3.2-1b" }) {
            DispatchQueue.main.async(execute: {
                model.size = Int(totalSize)
                // We'll use a special URL format to indicate we need to download multiple files
                model.downloadURL = "huggingface://\(self.defaultHuggingFaceRepo.owner)/\(self.defaultHuggingFaceRepo.repo)"
            })
        }
    }
    
    // MARK: - Public Methods
    
    public func getAvailableModels() -> [Model] {
        // Refresh download status for all models
        for model in availableModels {
            model.refreshDownloadStatus()
        }
        return availableModels
    }
    
    func getModelStoragePath() -> String {
        return modelStorageDirectory.path
    }
    
    /// Gets the full path to a specific model directory
    public func getModelPath(for modelId: String) -> URL {
        let modelsDirectory = getModelsDirectory()

        // Special case for default model - always use the same directory name
        if modelId == "llama-3.2-1b" {
            // Use the standard format for the default model
            let defaultModelPath = modelsDirectory.appendingPathComponent("llama_3_2_1b_iosv2_0")
            
            // Create the directory if it doesn't exist
            if !FileManager.default.fileExists(atPath: defaultModelPath.path) {
                do {
                    try FileManager.default.createDirectory(at: defaultModelPath, withIntermediateDirectories: true, attributes: nil)
                    print("Created default model directory at: \(defaultModelPath.path)")
                } catch {
                    print("Error creating default model directory: \(error)")
                }
            }
            
            return defaultModelPath
        }
        
        // For other models, use the sanitized model ID
        let sanitizedModelId = sanitizeModelId(modelId)
        let modelDir = modelsDirectory.appendingPathComponent(sanitizedModelId)

        // Create the directory if it doesn't exist
        if !FileManager.default.fileExists(atPath: modelDir.path) {
            do {
                try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true, attributes: nil)
            } catch {
                print("Error creating model directory: \(error)")
            }
        }

        return modelDir
    }
    
    private func sanitizeModelId(_ modelId: String) -> String {
        // Replace any characters that are not allowed in directory names
        let sanitized = modelId.replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "\\", with: "_")
            .replacingOccurrences(of: ":", with: "_")
            .replacingOccurrences(of: "*", with: "_")
            .replacingOccurrences(of: "?", with: "_")
            .replacingOccurrences(of: "\"", with: "_")
            .replacingOccurrences(of: "<", with: "_")
            .replacingOccurrences(of: ">", with: "_")
            .replacingOccurrences(of: "|", with: "_")
        print("Sanitized model ID: \(modelId) -> \(sanitized)")
        return sanitized
    }
    
    private func getModelsDirectory() -> URL {
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelsDirectory = documentsDirectory.appendingPathComponent("Models")
        print("Models directory path: \(modelsDirectory.path)")
        
        // Ensure the directory exists
        if !FileManager.default.fileExists(atPath: modelsDirectory.path) {
            do {
                try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true, attributes: nil)
                print("Created models directory")
            } catch {
                print("Error creating models directory: \(error)")
            }
        }
        
        return modelsDirectory
    }
    
    func getDeviceTypeString() -> String {
        switch deviceType {
        case .mac:
            return "Mac"
        case .macCatalyst:
            return "Mac Catalyst"
        case .iPad:
            return "iPad"
        case .iPhone:
            return "iPhone"
        case .other:
            return "Other"
        }
    }
    
    // Download a model by ID - preserves existing files to support resuming downloads
    public func downloadModel(modelId: String, fileProgress: @escaping (String, Double) -> Void, completion: @escaping (Bool) -> Void) {
        print("Starting download for model: \(modelId) (preserving existing files)")
        
        // Prevent device from sleeping during download
        preventSleepDuringDownload()
        
        // Find the model
        guard let model = getModel(for: modelId) else {
            print("Error: Model with ID \(modelId) not found")
            DispatchQueue.main.async {
                completion(false)
            }
            // Allow sleep if download fails immediately
            allowSleepAfterDownload()
            return
        }
        
        // Print the model's download URL for debugging
        print("Model download URL: \(model.downloadURL)")
        
        // IMPORTANT: Use the model's actual ID, not the modelId parameter
        // This ensures we're using the correct ID for custom models
        let actualModelId = model.id
        print("Using actual model ID for download: \(actualModelId)")
        
        // Fetch the latest model information before downloading
        if model.downloadURL.hasPrefix("https://huggingface.co/") || 
           model.downloadURL.hasPrefix("huggingface://") {
            print("Fetching latest model information before download...")
            fetchModelInformation()
        }
        
        print("Found model to download: \(actualModelId) - \(model.name)")
        
        // Check if we have write access before starting download
        let testFilePath = modelStorageDirectory.appendingPathComponent("write_test.txt")
        do {
            try "Test download access".write(to: testFilePath, atomically: true, encoding: .utf8)
            try fileManager.removeItem(at: testFilePath)
            print("Write access confirmed for download")
        } catch {
            print("ERROR: No write access to models directory for download: \(error)")
            DispatchQueue.main.async {
                completion(false)
            }
            // Allow sleep if we don't have write access
            allowSleepAfterDownload()
            return
        }
        
        // Get the proper model directory path - use actualModelId instead of modelId
        let modelDir = getModelPath(for: actualModelId)
        print("Model directory for download: \(modelDir.path)")
        
        // Create model directory if it doesn't exist (don't remove if it exists)
        if !fileManager.fileExists(atPath: modelDir.path) {
            do {
                try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true, attributes: nil)
                print("Created model directory at: \(modelDir.path)")
            } catch {
                print("ERROR: Failed to create model directory: \(error)")
                DispatchQueue.main.async {
                    completion(false)
                }
                return
            }
        } else {
            print("Model directory already exists at: \(modelDir.path), preserving existing files")
        }
        
        // Update model status to indicate download is starting
        model.isDownloaded = false
        
        // Store progress and completion handlers - use actualModelId
        // Note: Don't call fileProgress with "Downloading..." here - it overwrites
        // the detailed status set by URLSession delegate. Just store a no-op
        // since fileProgressObservers handles the actual progress display.
        progressObservers[actualModelId] = { _ in
            // Progress is handled by fileProgressObservers via URLSession delegate
            // Don't overwrite status here - it causes UI flashing
        }
        
        fileProgressObservers[actualModelId] = fileProgress
        completionHandlers[actualModelId] = { success in
            DispatchQueue.main.async {
                // Update model status
                if let model = self.getModel(for: actualModelId) {
                    // Verify model files
                    let isValid = self.verifyModelFiles(modelId: actualModelId)
                    print("Model verification result: \(isValid)")
                    
                    // Only set isDownloaded to true if verification passes
                    model.isDownloaded = isValid
                    
                    // Update model download status
                    self.modelDownloadStatus[actualModelId] = isValid ? .downloaded : 
                                                            (success ? .partiallyDownloaded : .failed)
                    
                    if success && isValid {
                        // Update downloaded models list if successful
                        if !self.downloadedModels.contains(where: { $0.id == model.id }) {
                            self.downloadedModels.append(model)
                            print("Added model to downloadedModels list: \(model.id)")
                        }
                        
                        // If this is the default model, update its ID mapping
                        if actualModelId == "llama-3.2-1b" {
                            // Update the model ID to match the directory name if needed
                            let dirName = modelDir.lastPathComponent
                            if dirName != "llama-3.2-1b" {
                                print("Updating default model ID mapping: llama-3.2-1b -> \(dirName)")
                                // You might want to store this mapping for future reference
                                UserDefaults.standard.set(dirName, forKey: "llama-3.2-1b-mapping")
                            }
                        }
                    } else if !isValid {
                        print("⚠️ Model verification failed - files may be missing or corrupted")
                        
                        // Remove from downloaded models list if it's there
                        self.downloadedModels.removeAll(where: { $0.id == model.id })
                        
                        // Log specific error message
                        if success {
                            print("❌ Download reported success but verification failed - check for missing files")
                        } else {
                            print("❌ Download failed and verification failed")
                        }
                        
                        // Save changes to ensure status persists
                        self.updateCustomModelsInUserDefaults()
                        
                        // Schedule cleanup of the download tracking after a delay
                        // This prevents the endless "Found download progress" messages
                        self.cleanupDownloadTracking(for: modelId, delay: 5.0)
                        
                        self.completionHandlers[modelId]?(false)
                    }
                    
                    // Call completion handler with verification result, not just download success
                    // This ensures the UI updates correctly when files are missing
                    completion(success && isValid)
                } else {
                    // Model not found, just return the success status
                completion(success)
                }
                
                // Clean up
                self.cleanupDownload(for: actualModelId)
                
                // Allow sleep after download completes
                self.allowSleepAfterDownload()
            }
        }
        
        // Update model download status to downloading
        modelDownloadStatus[actualModelId] = .downloading

        // Check if this is a Hugging Face model
        // Supported formats:
        // 1. https://huggingface.co/owner/repo
        // 2. huggingface://owner/repo
        // 3. owner/repo (shorthand - must contain exactly one "/" and no other URL parts)
        let isHuggingFaceURL = model.downloadURL.hasPrefix("https://huggingface.co/") ||
                               model.downloadURL.hasPrefix("huggingface://")

        // Check for shorthand format: "owner/repo" (contains one slash, no protocol, no dots except in repo name)
        let isHuggingFaceShorthand: Bool = {
            let url = model.downloadURL
            // Must contain exactly one "/"
            let slashCount = url.filter { $0 == "/" }.count
            guard slashCount == 1 else { return false }
            // Must not be an actual URL (no protocol)
            guard !url.contains("://") else { return false }
            // Split and verify both parts exist
            let parts = url.split(separator: "/")
            guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else { return false }
            // First part (owner) should not contain dots (to avoid domain.com/path)
            guard !parts[0].contains(".") else { return false }
            return true
        }()

        if isHuggingFaceURL || isHuggingFaceShorthand {
            print("Starting Hugging Face model download (detected format: \(isHuggingFaceURL ? "full URL" : "shorthand"))")
            // Convert shorthand to full URL if needed
            let fullModelURL: String
            if isHuggingFaceShorthand {
                fullModelURL = "https://huggingface.co/\(model.downloadURL)"
                print("Converted shorthand '\(model.downloadURL)' to full URL: \(fullModelURL)")
            } else {
                fullModelURL = model.downloadURL
            }
            downloadHuggingFaceModel(modelId: actualModelId, modelURL: fullModelURL)
            return
        }

        // For regular URL downloads
        guard let url = URL(string: model.downloadURL) else {
            print("Invalid download URL: \(model.downloadURL)")
            setDownloadError(for: actualModelId, message: "Invalid download URL format")
            DispatchQueue.main.async {
                completion(false)
            }
            return
        }

        print("Starting download from URL: \(url)")
        
        // Create download task
        let downloadTask = URLSession.shared.downloadTask(with: url)
        downloadTasks[actualModelId] = downloadTask
        downloadTask.resume()
    }
    
    private func downloadHuggingFaceModel(modelId: String, modelURL: String) {
        print("\nStarting Hugging Face model download: \(modelId) from URL: \(modelURL)")
        print("Note: This download will preserve existing files rather than doing a clean download")

        // Extract repository information from the URL
        guard let repoInfo = extractHuggingFaceRepoInfo(from: modelURL) else {
            print("Failed to extract repository information from URL: \(modelURL)")
            reportDownloadFailure(for: modelId)
            return
        }

        print("Repository information:")
        print("- Owner: \(repoInfo.owner)")
        print("- Repository: \(repoInfo.repo)")

        // Create HuggingFaceRepo object for download
        let huggingFaceRepo = HuggingFaceRepo(
            owner: repoInfo.owner,
            repo: repoInfo.repo,
            branch: "main"
        )

        // Create directory for the model but don't remove if it exists
        let modelDir = getModelPath(for: modelId)
        do {
            if !fileManager.fileExists(atPath: modelDir.path) {
                try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)
                print("Created model directory: \(modelDir.path)")
            } else {
                print("Using existing model directory: \(modelDir.path)")
            }
        } catch {
            print("Error creating model directory: \(error)")
            self.reportDownloadFailure(for: modelId)
            return
        }

        // Update UI to show we're starting and record start time for ETA
        DispatchQueue.main.async {
            self.currentDownloadingFiles[modelId] = "Downloading model configuration..."
            self.progressObservers[modelId]?(0.02)
        }
        downloadStartTimes[modelId] = Date()
        lastProgressUpdateTime[modelId] = Date()

        // Reset progress smoothing values for a fresh download
        maxDownloadedBytes[modelId] = 0
        maxDownloadProgress[modelId] = 0.0

        // Start download watchdog timer to detect stalls
        startDownloadWatchdog(for: modelId)
        logDownload("▶️ STARTED: \(modelId)")

        // First, download meta.yaml to get the correct capitalization and model structure
        // Using NON-BLOCKING approach to keep UI responsive
        print("Downloading meta.yaml first to get correct capitalization...")
        let metaYamlURL = URL(string: "https://huggingface.co/\(repoInfo.owner)/\(repoInfo.repo)/resolve/main/meta.yaml")!
        let metaYamlPath = modelDir.appendingPathComponent("meta.yaml")

        print("📥 Starting meta.yaml download from: \(metaYamlURL)")
        print("📂 Saving to: \(metaYamlPath.path)")

        // Use non-blocking async download instead of semaphore
        downloadMetaYamlAsync(
            url: metaYamlURL,
            destination: metaYamlPath,
            modelId: modelId
        ) { [weak self] success in
            guard let self = self else { return }

            if !success {
                print("❌ meta.yaml download failed")
                self.setDownloadError(for: modelId, message: "Failed to download model configuration (meta.yaml)")
                self.reportDownloadFailure(for: modelId)
                return
            }

            // Parse meta.yaml and continue with model files download
            self.continueHuggingFaceDownload(
                modelId: modelId,
                metaYamlPath: metaYamlPath,
                modelDir: modelDir,
                huggingFaceRepo: huggingFaceRepo
            )
        }
    }

    /// Downloads meta.yaml asynchronously without blocking
    private func downloadMetaYamlAsync(url: URL, destination: URL, modelId: String, completion: @escaping (Bool) -> Void) {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        let session = URLSession(configuration: config)

        let task = session.dataTask(with: url) { [weak self] data, response, error in
            if let error = error {
                print("❌ Error downloading meta.yaml: \(error.localizedDescription)")
                self?.setDownloadError(for: modelId, message: "Network error: \(error.localizedDescription)")
                DispatchQueue.main.async { completion(false) }
                return
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                print("❌ Invalid response type for meta.yaml")
                self?.setDownloadError(for: modelId, message: "Invalid server response")
                DispatchQueue.main.async { completion(false) }
                return
            }

            if httpResponse.statusCode != 200 {
                print("❌ HTTP error for meta.yaml: \(httpResponse.statusCode)")
                self?.setDownloadError(for: modelId, message: "Server error: HTTP \(httpResponse.statusCode)")
                DispatchQueue.main.async { completion(false) }
                return
            }

            guard let data = data, !data.isEmpty else {
                print("❌ Empty data received for meta.yaml")
                self?.setDownloadError(for: modelId, message: "Empty configuration file received")
                DispatchQueue.main.async { completion(false) }
                return
            }

            print("✅ meta.yaml download successful: \(data.count) bytes")

            // Write the file
            do {
                try data.write(to: destination)
                print("✅ meta.yaml saved successfully to \(destination.path)")

                // Print the content for debugging
                if let content = String(data: data, encoding: .utf8) {
                    print("📃 meta.yaml content:\n\(content)")
                }

                DispatchQueue.main.async { completion(true) }
            } catch {
                print("❌ Error saving meta.yaml: \(error.localizedDescription)")
                self?.setDownloadError(for: modelId, message: "Failed to save configuration: \(error.localizedDescription)")
                DispatchQueue.main.async { completion(false) }
            }
        }

        task.resume()
    }

    /// Continues the HuggingFace download after meta.yaml is downloaded
    private func continueHuggingFaceDownload(modelId: String, metaYamlPath: URL, modelDir: URL, huggingFaceRepo: HuggingFaceRepo) {
        // Read and parse meta.yaml content
        do {
            let metaYamlContent = try String(contentsOf: metaYamlPath, encoding: .utf8)

            // Confirm we have non-empty content
            guard !metaYamlContent.isEmpty else {
                print("❌ Empty meta.yaml content")
                self.setDownloadError(for: modelId, message: "Empty model configuration")
                self.reportDownloadFailure(for: modelId)
                return
            }

            print("📝 Parsing meta.yaml content...")

            // Parse meta.yaml to get model configuration
            let config = try ModelConfiguration(from: metaYamlContent, modelPath: modelDir.path)
            print("📋 Model configuration parsed successfully")

            // Initialize collections for blobs and trees information
            let blobs: [[String: Any]] = []
            let trees: [[String: Any]] = []

            DispatchQueue.main.async {
                self.progressObservers[modelId]?(0.1) // 10% for downloading meta.yaml and parsing
                self.currentDownloadingFiles[modelId] = "Analyzed model structure..."
            }

            // Get the required files based on the configuration
            let requiredFiles = self.getRequiredFiles(from: config)
            
            // Log required files for debugging
            print("📋 Required files determined from meta.yaml (\(requiredFiles.count) files):")
            for (index, file) in requiredFiles.prefix(5).enumerated() {
                print("  \(index+1). \(file)")
            }
            if requiredFiles.count > 5 {
                print("  ... and \(requiredFiles.count - 5) more files")
            }
            
            // Download required files from the repository with improved handling
            self.downloadModelFiles(
                modelId: modelId,
                requiredFiles: requiredFiles,
                modelDir: modelDir,
                blobs: blobs,
                trees: trees,
                config: config,
                huggingFaceRepo: huggingFaceRepo
            )
            
        } catch {
            print("❌ Error processing meta.yaml: \(error)")
            DispatchQueue.main.async {
                self.completionHandlers[modelId]?(false)
            }
        }
    }
    
    private func downloadHuggingFaceFiles(
        modelId: String, 
        effectiveModelId: String, 
        files: [[String: Any]], 
        blobs: [[String: Any]],
        trees: [[String: Any]],
        huggingFaceRepo: HuggingFaceRepo
    ) {
        // IMPORTANT: For the default model ("llama-3.2-1b"), we need to use the original modelId
        // to ensure files are downloaded to the hardcoded path "llama_3_2_1b_iosv2_0"
        // rather than using effectiveModelId which might be derived from the repository name.
        
        // Get the model directory path - use the original modelId for the default model, otherwise use effectiveModelId
        let originalModelId = modelId
        let modelDir = getModelPath(for: originalModelId == "llama-3.2-1b" ? originalModelId : effectiveModelId)
        print("Preparing model directory for download: \(modelDir.path)")
        
        // Start activity indicator for this download
        startDownloadActivityIndicator(for: modelId)
        
        // Ensure model directory exists but don't remove existing files
        do {
            if !fileManager.fileExists(atPath: modelDir.path) {
                try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)
                print("Created model directory: \(modelDir.path)")
            } else {
                print("Using existing model directory (preserving files): \(modelDir.path)")
            }
        } catch {
            print("Error ensuring model directory exists: \(error)")
            reportDownloadFailure(for: modelId)
            return
        }
        
        // Step 1: Find and download meta.yaml first
        guard files.first(where: { ($0["path"] as? String) == "meta.yaml" }) != nil else {
            print("Error: meta.yaml not found in repository")
            DispatchQueue.main.async {
                self.completionHandlers[modelId]?(false)
            }
            return
        }

        print("Step 1: Downloading meta.yaml...")
        let metaYamlURL = URL(string: "\(huggingFaceRepo.downloadBaseUrl)meta.yaml")!
        let metaYamlDestination = modelDir.appendingPathComponent("meta.yaml")

        // Add detailed logging for debugging custom model issues
        print("🔍 DEBUG - Meta YAML download details:")
        print("  - Full URL: \(metaYamlURL.absoluteString)")
        print("  - Base URL: \(huggingFaceRepo.downloadBaseUrl)")
        print("  - Repository owner: \(huggingFaceRepo.owner)")
        print("  - Repository name: \(huggingFaceRepo.repo)")
        print("  - Branch: \(huggingFaceRepo.branch)")
        print("  - Destination: \(metaYamlDestination.path)")

        // Store in instance property to prevent premature deallocation (semaphore crash)
        let downloadGroup = DispatchGroup()
        withDownloadLock { activeDownloadGroups["\(modelId):meta"] = downloadGroup }
        downloadGroup.enter()

        let downloadTask = URLSession.shared.downloadTask(with: metaYamlURL) { [weak self] tempURL, response, error in
            defer {
                downloadGroup.leave()
                // DELAYED cleanup - wait for internal semaphore operations to complete
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    self?.withDownloadLock { _ = self?.activeDownloadGroups.removeValue(forKey: "\(modelId):meta") }
                }
            }
            guard let self = self else { return }
            
            if let error = error {
                print("Error downloading meta.yaml: \(error)")
                DispatchQueue.main.async {
                    self.completionHandlers[modelId]?(false)
                }
                return
            }
            
            guard let tempURL = tempURL else {
                print("Error: No data received for meta.yaml")
                DispatchQueue.main.async {
                    self.completionHandlers[modelId]?(false)
                }
                return
            }
            
            do {
                try self.fileManager.moveItem(at: tempURL, to: metaYamlDestination)
                print("Successfully downloaded meta.yaml to \(metaYamlDestination.path)")
                
                // Print meta.yaml content immediately after download
                let metaYamlContent = try String(contentsOf: metaYamlDestination, encoding: .utf8)
                print("\n📝 META.YAML CONTENT:")
                print(metaYamlContent)
                print("END OF META.YAML\n")
                
                // Update status to indicate download is officially starting
                print("🚀 Starting model download for \(modelId)...")
                self.currentDownloadingFiles[modelId] = "Starting model download..."
                
                // Step 2: Parse meta.yaml to create ModelConfiguration
                let config = try ModelConfiguration(from: metaYamlContent, modelPath: modelDir.path)
                
                // Step 3: Get required files using the improved method
                let requiredFiles = self.getRequiredFiles(from: config)
                
                print("Required files determined from meta.yaml:")
                requiredFiles.forEach { print("- \($0)") }
                
                // Step 4: Download required files from the repository, handling tree and blob structures
                self.downloadModelFiles(
                    modelId: modelId,
                    requiredFiles: requiredFiles,
                    modelDir: modelDir,
                    blobs: blobs,
                    trees: trees,
                    config: config,
                    huggingFaceRepo: huggingFaceRepo
                )
                
            } catch {
                print("Error processing meta.yaml: \(error)")
                DispatchQueue.main.async {
                    self.completionHandlers[modelId]?(false)
                }
            }
        }

        downloadTask.resume()
    }

    // Updated method to download files, handling tree and blob structures
    private func downloadModelFiles(
        modelId: String, 
        requiredFiles: [String],
        modelDir: URL, 
        blobs: [[String: Any]],
        trees: [[String: Any]],
        config: ModelConfiguration, 
        huggingFaceRepo: HuggingFaceRepo
    ) {
        print("📥 Starting download of model files: \(requiredFiles.count) files required")
        
        // Copy the required files to ensure thread safety
        var sortedFiles = requiredFiles.sorted()
        
        // Check for .mlmodelc directories and add weight.bin path if needed
        // BUT avoid duplicates - getRequiredFiles already adds individual files including weights/weight.bin
        var expandedFiles: [String] = []
        var addedPaths = Set<String>()  // Track what we've added to avoid duplicates

        for file in sortedFiles {
            if file.hasSuffix(".mlmodelc") && !file.contains("/") {
                // This is a bare directory name (e.g., "xxx.mlmodelc") not a path within it
                // Convert to weight.bin path, but only if not already in list
                let weightBinPath = "\(file)/weights/weight.bin"
                if !addedPaths.contains(weightBinPath) && !sortedFiles.contains(weightBinPath) {
                    print("🔄 Converting directory to weight file: \(weightBinPath)")
                    expandedFiles.append(weightBinPath)
                    addedPaths.insert(weightBinPath)
                } else {
                    print("⏭️ Skipping duplicate directory expansion: \(weightBinPath)")
                }
            } else if !addedPaths.contains(file) {
                expandedFiles.append(file)
                addedPaths.insert(file)
            } else {
                print("⏭️ Skipping duplicate file: \(file)")
            }
        }

        // Replace the sorted files with our deduplicated list
        sortedFiles = expandedFiles
        print("📊 After deduplication: \(sortedFiles.count) files (was \(requiredFiles.count))")

        // IMPORTANT: Filter out files that are already fully downloaded
        // This enables resume functionality - only download missing/incomplete files
        var filesToDownload: [String] = []
        var alreadyDownloadedBytes: Int64 = 0
        var skippedCount = 0

        for file in sortedFiles {
            let fileURL = modelDir.appendingPathComponent(file)
            if fileManager.fileExists(atPath: fileURL.path) {
                // File exists - check if it has content (not empty)
                do {
                    let attributes = try fileManager.attributesOfItem(atPath: fileURL.path)
                    if let fileSize = attributes[.size] as? Int64, fileSize > 0 {
                        // File exists and has content - skip download
                        alreadyDownloadedBytes += fileSize
                        skippedCount += 1
                        logDownload("⏭️ Skipping already downloaded: \(file) (\(fileSize) bytes)")
                        continue
                    }
                } catch {
                    // Can't read attributes, download anyway
                    logDownload("⚠️ Can't check file, will re-download: \(file)")
                }
            }
            filesToDownload.append(file)
        }

        if skippedCount > 0 {
            logDownload("📊 Resuming download: \(skippedCount) files already downloaded, \(filesToDownload.count) remaining")
            // Update progress tracking to account for already downloaded bytes
            withDownloadLock {
                bytesDownloadedPerFile["\(modelId):_already_downloaded"] = alreadyDownloadedBytes
            }
        }

        // Use filtered list (only files that need downloading)
        sortedFiles = filesToDownload
        let totalFiles = sortedFiles.count

        // If all files are already downloaded, report success immediately
        if totalFiles == 0 {
            logDownload("✅ All files already downloaded for \(modelId)")
            DispatchQueue.main.async {
                self.progressObservers[modelId]?(1.0)
                self.currentDownloadingFiles[modelId] = "All files already downloaded!"
                self.completionHandlers[modelId]?(true)
            }
            return
        }

        let maxConcurrentDownloads = 2  // Limit concurrent downloads to avoid network issues

        // Use a dispatch group to wait for all downloads to complete
        // Store ALL synchronization primitives in instance properties to prevent premature deallocation

        // IMPORTANT: Clean up any existing GCD objects from a previous cancelled download
        // This prevents crashes when cancel -> restart quickly
        withDownloadLock {
            if activeDownloadGroups["\(modelId):files"] != nil {
                print("⚠️ Cleaning up existing download group for \(modelId)")
                _ = activeDownloadGroups.removeValue(forKey: "\(modelId):files")
            }
            if activeDownloadSemaphores["\(modelId):concurrent"] != nil {
                print("⚠️ Cleaning up existing semaphore for \(modelId)")
                _ = activeDownloadSemaphores.removeValue(forKey: "\(modelId):concurrent")
            }
            if activeDownloadQueues["\(modelId):counter"] != nil {
                print("⚠️ Cleaning up existing counter queue for \(modelId)")
                _ = activeDownloadQueues.removeValue(forKey: "\(modelId):counter")
            }
            if activeDownloadQueues["\(modelId):download"] != nil {
                print("⚠️ Cleaning up existing download queue for \(modelId)")
                _ = activeDownloadQueues.removeValue(forKey: "\(modelId):download")
            }
        }

        let downloadGroup = DispatchGroup()
        let counterQueue = DispatchQueue(label: "com.anemll.modelfiledownload.\(modelId).\(Date().timeIntervalSince1970)")
        let downloadQueue = DispatchQueue(label: "com.anemll.modelfiledownload.\(modelId).concurrent.\(Date().timeIntervalSince1970)", attributes: .concurrent)
        withDownloadLock {
            activeDownloadGroups["\(modelId):files"] = downloadGroup
            activeDownloadQueues["\(modelId):counter"] = counterQueue
            activeDownloadQueues["\(modelId):download"] = downloadQueue
        }

        print("📂 Model directory: \(modelDir.path)")

        // Use a semaphore to limit concurrent downloads
        // Store in instance property to prevent premature deallocation (semaphore crash)
        let semaphore = DispatchSemaphore(value: maxConcurrentDownloads)
        withDownloadLock { activeDownloadSemaphores["\(modelId):concurrent"] = semaphore }

        // Update UI to show we're calculating total size first
        DispatchQueue.main.async {
            self.progressObservers[modelId]?(0.05)  // Start at 5% (meta.yaml done, calculating sizes)
            self.currentDownloadingFiles[modelId] = "Calculating total download size..."
        }

        // DEBUG: Print all files that will be downloaded
        print("📋 ALL FILES TO DOWNLOAD (\(totalFiles) total):")
        for (index, file) in sortedFiles.enumerated() {
            print("   [\(index + 1)/\(totalFiles)] \(file)")
        }

        // First, fetch file sizes from HuggingFace API to calculate total expected bytes
        calculateTotalExpectedBytes(for: modelId, files: sortedFiles, owner: huggingFaceRepo.owner, repo: huggingFaceRepo.repo) { [weak self] totalBytes in
            guard let self = self else { return }

            // Update UI to show total size calculated
            DispatchQueue.main.async {
                let totalMB = Double(totalBytes) / 1_000_000.0
                self.progressObservers[modelId]?(0.1)  // 10% progress after calculating sizes
                self.currentDownloadingFiles[modelId] = String(format: "Starting download (%.1f MB total)...", totalMB)
            }

            // Now start the actual downloads
            self.startFileDownloads(
                modelId: modelId,
                sortedFiles: sortedFiles,
                totalFiles: totalFiles,
                modelDir: modelDir,
                huggingFaceRepo: huggingFaceRepo,
                downloadGroup: downloadGroup,
                downloadQueue: downloadQueue,
                counterQueue: counterQueue,
                semaphore: semaphore
            )
        }
    }

    /// Internal method to start file downloads after total size is calculated
    private func startFileDownloads(
        modelId: String,
        sortedFiles: [String],
        totalFiles: Int,
        modelDir: URL,
        huggingFaceRepo: HuggingFaceRepo,
        downloadGroup: DispatchGroup,
        downloadQueue: DispatchQueue,
        counterQueue: DispatchQueue,
        semaphore: DispatchSemaphore
    ) {
        // Initialize counters for download progress tracking
        var completedFiles = 0
        var failedFiles = 0
        var inProgressDownloads = 0

        // Track overall success
        var overallSuccess = true

        // Track enter/leave balance for debugging
        var enterCount = 0
        var leaveCount = 0
        var pendingFiles: Set<Int> = []  // Track which file indices are still pending
        let balanceQueue = DispatchQueue(label: "com.anemll.balance.\(modelId)")

        // Process each required file
        for (fileIndex, requiredFile) in sortedFiles.enumerated() {
            // Enter the dispatch group for this download
            downloadGroup.enter()
            balanceQueue.sync {
                enterCount += 1
                pendingFiles.insert(fileIndex + 1)
                print("⬆️ ENTER [\(enterCount)] for file [\(fileIndex + 1)/\(totalFiles)]: \(requiredFile)")
            }

            // Capture file info for closure
            let capturedFile = requiredFile
            let capturedIndex = fileIndex

            // Log when async block is queued
            print("📤 QUEUED async block for file [\(capturedIndex + 1)/\(totalFiles)]: \(capturedFile)")

            // Wait for a semaphore slot to become available
            downloadQueue.async {
                print("🚀 ASYNC BLOCK STARTED for file [\(capturedIndex + 1)/\(totalFiles)]: \(capturedFile)")
                print("🔄 Waiting for semaphore slot for file [\(capturedIndex + 1)/\(totalFiles)]: \(capturedFile)")
                semaphore.wait()
                print("✅ Got semaphore slot for file [\(capturedIndex + 1)/\(totalFiles)]: \(capturedFile)")
                
                // Create the full URL for the file download
                let fileURLString = "https://huggingface.co/\(huggingFaceRepo.owner)/\(huggingFaceRepo.repo)/resolve/main/\(capturedFile)"
                let destinationURL = modelDir.appendingPathComponent(capturedFile)

                // Update counter for in-progress downloads
                counterQueue.sync {
                    inProgressDownloads += 1

                    // Note: Real-time byte-based progress is shown via handleFileDownloadProgress
                    // This just tracks which file we're starting to download
                    let fileName = (capturedFile as NSString).lastPathComponent
                    print("📥 Starting download of \(fileName) (\(completedFiles+1)/\(totalFiles)) [index \(capturedIndex + 1)]")
                }

                // Track if this file's completion has been called (to prevent double-call from timeout)
                var fileCompleted = false
                let fileCompletionLock = NSLock()

                // Completion handler wrapper that ensures single execution
                let handleFileCompletion: (Bool) -> Void = { success in
                    fileCompletionLock.lock()
                    if fileCompleted {
                        fileCompletionLock.unlock()
                        print("⚠️ Ignoring duplicate completion for file [\(capturedIndex + 1)/\(totalFiles)]: \(capturedFile)")
                        return
                    }
                    fileCompleted = true
                    fileCompletionLock.unlock()

                    // Update counters based on result
                    counterQueue.sync {
                        completedFiles += 1
                        inProgressDownloads -= 1

                        if !success {
                            failedFiles += 1
                            overallSuccess = false
                            print("❌ Failed to download: \(capturedFile)")
                        }

                        // Only update status message on errors - byte-based progress shows real-time updates
                        if !success {
                            let fileName = (capturedFile as NSString).lastPathComponent
                            let status = "Error downloading: \(fileName) (\(completedFiles)/\(totalFiles))"
                            DispatchQueue.main.async {
                                self.currentDownloadingFiles[modelId] = status
                            }
                        }

                        // Log completion for debugging
                        print("📦 File completed (\(completedFiles)/\(totalFiles)): \(capturedFile) [index \(capturedIndex + 1)]")
                    }

                    // Release the semaphore slot for next download
                    print("🔓 Signaling semaphore for file [\(capturedIndex + 1)/\(totalFiles)]: \(capturedFile)")
                    semaphore.signal()

                    // Mark this download as complete in the group
                    balanceQueue.sync {
                        leaveCount += 1
                        pendingFiles.remove(capturedIndex + 1)
                        print("⬇️ LEAVE [\(leaveCount)] for file [\(capturedIndex + 1)/\(totalFiles)]: \(capturedFile) (balance: enter=\(enterCount), leave=\(leaveCount))")
                        if leaveCount < enterCount {
                            print("   📌 Still pending: \(pendingFiles.sorted())")
                        }
                    }
                    downloadGroup.leave()
                }

                // Set up progress-aware stall detection
                // Instead of a fixed timeout from start, check if progress is being made
                // This allows large files to take as long as needed if they're actively downloading
                let stallCheckInterval: Double = 30.0  // Check every 30 seconds
                let maxStallTime: Double = 90.0  // Fail if no progress for 90 seconds

                // Use a timer that repeatedly checks for stalls
                var lastProgressCheck = Date()
                var lastBytesForFile: Int64 = 0

                let stallCheckTimer = Timer.scheduledTimer(withTimeInterval: stallCheckInterval, repeats: true) { [weak self] timer in
                    guard let self = self else {
                        timer.invalidate()
                        return
                    }

                    fileCompletionLock.lock()
                    let alreadyCompleted = fileCompleted
                    fileCompletionLock.unlock()

                    if alreadyCompleted {
                        timer.invalidate()
                        return
                    }

                    // Check if progress was made since last check
                    let fileKey = "\(modelId):\(capturedFile)"
                    let currentBytes = self.bytesDownloadedPerFile[fileKey] ?? 0

                    if currentBytes > lastBytesForFile {
                        // Progress is being made, reset stall timer
                        lastProgressCheck = Date()
                        lastBytesForFile = currentBytes
                        let downloadedMB = Double(currentBytes) / 1_000_000.0
                        self.logDownload("📊 Progress: \(capturedFile) - \(String(format: "%.1f", downloadedMB)) MB")
                    } else {
                        // No progress, check if stalled too long
                        let stallTime = Date().timeIntervalSince(lastProgressCheck)
                        if stallTime > maxStallTime {
                            self.logDownload("⏰ STALLED: \(capturedFile) - no progress for \(Int(stallTime))s")
                            timer.invalidate()
                            handleFileCompletion(false)
                        } else {
                            self.logDownload("⚠️ Slow: \(capturedFile) - no progress for \(Int(stallTime))s")
                        }
                    }
                }

                // Make sure timer runs on main run loop
                RunLoop.main.add(stallCheckTimer, forMode: .common)

                // Download the file with streaming support and progress tracking
                self.downloadSingleFile(urlString: fileURLString, destination: destinationURL, modelId: modelId) { success in
                    handleFileCompletion(success)
                }
            }
        }
        
        // Add a timeout diagnostic - if downloads stall, print which files are stuck
        // Use longer timeout since weight files can take several minutes
        DispatchQueue.main.asyncAfter(deadline: .now() + 300.0) {  // 5 minutes
            balanceQueue.sync {
                if leaveCount < enterCount {
                    print("[DOWNLOAD] ⏰ DIAGNOSTIC: Downloads may be stuck after 5 minutes")
                    print("   Enter count: \(enterCount), Leave count: \(leaveCount)")
                    print("   🚨 STILL PENDING (indices): \(pendingFiles.sorted())")
                    // Print file names for stuck indices
                    for idx in pendingFiles.sorted() {
                        if idx > 0 && idx <= sortedFiles.count {
                            print("      [\(idx)/\(totalFiles)] \(sortedFiles[idx - 1])")
                        }
                    }
                }
            }
        }

        // Use notify() instead of wait() to avoid blocking the download queue
        // wait() can cause deadlock if called on the same queue where downloads run
        print("🔔 Setting up downloadGroup.notify() - will fire when all \(totalFiles) files complete")
        downloadGroup.notify(queue: .main) { [weak self] in
            print("🎉 downloadGroup.notify() FIRED! All downloads complete.")
            guard let self = self else {
                print("❌ self is nil in notify callback!")
                return
            }

            // DELAYED cleanup - wait for internal semaphore operations to complete
            // This prevents "Semaphore object deallocated while in use" crash
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                self.withDownloadLock {
                    _ = self.activeDownloadGroups.removeValue(forKey: "\(modelId):files")
                    _ = self.activeDownloadSemaphores.removeValue(forKey: "\(modelId):concurrent")
                    _ = self.activeDownloadQueues.removeValue(forKey: "\(modelId):counter")
                    _ = self.activeDownloadQueues.removeValue(forKey: "\(modelId):download")
                }
            }

            // Print download summary
            print("📊 Download summary:")
            print("  - Total files: \(totalFiles)")
            print("  - Completed: \(completedFiles)")
            print("  - Failed: \(failedFiles)")
            print("  - Overall success: \(overallSuccess)")

            // Update UI with final status (already on main queue from notify)
            if overallSuccess {
                print("✅ Download successful, starting verification...")
                self.currentDownloadingFiles[modelId] = "Download complete. Verifying files..."

                // Verify the model files
                let isValid = self.verifyModelFiles(modelId: modelId)
                print("🔍 Model verification result: \(isValid)")

                if isValid {
                    print("✅ Verification passed, updating model status...")
                    // Update model status to downloaded
                    self.modelDownloadStatus[modelId] = .downloaded

                    // Update model in available models list
                    if let model = self.availableModels.first(where: { $0.id == modelId }) {
                        model.isDownloaded = true
                        print("✅ Model marked as downloaded: \(model.name)")

                        // Add to downloaded models list if not already there
                        if !self.downloadedModels.contains(where: { $0.id == model.id }) {
                            self.downloadedModels.append(model)
                        }
                    }

                    // Call completion handler with success
                    self.logDownload("✅ SUCCESS: Download completed for \(modelId)")
                    self.stopDownloadWatchdog(for: modelId)
                    self.completionHandlers[modelId]?(true)
                } else {
                    // Model verification failed - likely missing weight files or other critical components
                    self.logDownload("⚠️ INCOMPLETE: Download finished but verification failed for \(modelId)")

                    // Check if weight files are missing (the most common issue)
                    let modelDir = self.getModelPath(for: modelId)
                    let missingWeightFiles = self.checkForMissingWeightFiles(in: modelDir)

                    if !missingWeightFiles.isEmpty {
                        print("❌ Missing critical weight files:")
                        for file in missingWeightFiles {
                            print("   - \(file)")
                        }
                        print("⚠️ This may indicate file availability issues with the repository")
                    }

                    self.modelDownloadStatus[modelId] = .partiallyDownloaded
                    self.currentDownloadingFiles[modelId] = "Download incomplete. Missing essential files."

                    // Mark model as not downloaded since verification failed
                    if let model = self.availableModels.first(where: { $0.id == modelId }) {
                        model.isDownloaded = false
                    }

                    // Save changes to ensure status persists
                    self.updateCustomModelsInUserDefaults()

                    // Schedule cleanup of the download tracking after a delay
                    self.cleanupDownloadTracking(for: modelId, delay: 5.0)

                    self.logDownload("❌ FAILED: Verification failed for \(modelId) - missing essential files")
                    self.stopDownloadWatchdog(for: modelId)
                    self.completionHandlers[modelId]?(false)
                }
            } else {
                // Download failed
                self.logDownload("❌ FAILED: Download failed with \(failedFiles) file errors for \(modelId)")
                self.modelDownloadStatus[modelId] = .failed
                self.currentDownloadingFiles[modelId] = "Download failed with \(failedFiles) file errors."

                // Schedule cleanup of the download tracking after a delay
                self.cleanupDownloadTracking(for: modelId, delay: 5.0)

                self.stopDownloadWatchdog(for: modelId)
                self.completionHandlers[modelId]?(false)
            }

            // Clean up resources
            print("🧹 Cleaning up download resources...")
            self.cleanupDownload(for: modelId)
            print("✅ Download finalization complete for modelId: \(modelId)")
        }
    }
    
    // Helper method to list all files recursively for debugging
    private func listAllFilesRecursively(at path: String, indent: String = "") {
        do {
            let items = try fileManager.contentsOfDirectory(atPath: path)
            for item in items {
                let itemPath = (path as NSString).appendingPathComponent(item)
                var isDir: ObjCBool = false
                if fileManager.fileExists(atPath: itemPath, isDirectory: &isDir) {
                    print("\(indent)📄 \(item)")
                    if isDir.boolValue {
                        listAllFilesRecursively(at: itemPath, indent: indent + "  ")
                    }
                }
            }
        } catch {
            print("Error listing files at \(path): \(error)")
        }
    }
    
    // Helper method to calculate weighted progress with weight.bin files accounting for 96% of .mlmodelc size
    private func calculateWeightedProgress(completedFiles: [String], totalFiles: [String]) -> Double {
        // Define weights for different file types
        let weightBinWeight = 0.96  // 96% of .mlmodelc size
        let otherFilesWeight = 0.04 / 4.0  // Remaining 4% divided equally among other files
        
        let totalWeight = totalFiles.reduce(0.0) { (result, file) -> Double in
            if file.contains("/weights/weight.bin") {
                return result + weightBinWeight
            } else if file.contains(".mlmodelc/") {
                return result + otherFilesWeight
            } else {
                return result + 0.01  // Non-mlmodelc files have small weight
            }
        }
        
        let completedWeight = completedFiles.reduce(0.0) { (result, file) -> Double in
            if file.contains("/weights/weight.bin") {
                return result + weightBinWeight
            } else if file.contains(".mlmodelc/") {
                return result + otherFilesWeight
            } else {
                return result + 0.01  // Non-mlmodelc files have small weight
            }
        }
        
        return completedWeight / totalWeight
    }
    
    // MARK: - URLSessionDownloadDelegate

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let taskId = downloadTask.taskIdentifier

        // Thread-safe access to taskToFileInfo
        let fileInfo: (modelId: String, fileName: String, destination: URL)? = withDownloadLock {
            return taskToFileInfo[taskId]
        }

        // First check if this is an individual file download (new system)
        if let fileInfo = fileInfo {
            handleFileDownloadCompletion(taskId: taskId, fileInfo: fileInfo, location: location)
            return
        }

        // Legacy handling for top-level model downloads
        guard let modelId = downloadTasks.first(where: {
            if let task = $0.value as? URLSessionDownloadTask {
                return task == downloadTask
            } else if let combinedTask = $0.value as? CombinedDownloadTask {
                return combinedTask.tasks.contains(downloadTask)
            }
            return false
        })?.key else {
            print("Could not find model ID for completed download task (taskId: \(taskId))")
            return
        }

        // Get the destination path for the model
        let destinationURL = modelStorageDirectory.appendingPathComponent(modelId)

        do {
            // Create model directory if it doesn't exist
            if !fileManager.fileExists(atPath: destinationURL.path) {
                try fileManager.createDirectory(at: destinationURL, withIntermediateDirectories: true)
            }

            // For single-file downloads (like .zip), move to the model directory
            let finalZipURL = destinationURL.appendingPathComponent("model.zip")
            try fileManager.moveItem(at: location, to: finalZipURL)

            print("Downloaded file saved to: \(finalZipURL.path)")

            // Update model status
            if let model = availableModels.first(where: { $0.id == modelId }) {
                model.isDownloaded = true

                // Update downloaded models list
                DispatchQueue.main.async {
                    if !self.downloadedModels.contains(where: { $0.id == model.id }) {
                        self.downloadedModels.append(model)
                    }
                }
            }

            DispatchQueue.main.async {
                self.modelDownloadStatus[modelId] = .downloaded
                self.completionHandlers[modelId]?(true)
            }
        } catch {
            print("Error saving downloaded file: \(error)")
            DispatchQueue.main.async {
                self.modelDownloadStatus[modelId] = .failed
                self.completionHandlers[modelId]?(false)
            }
        }
    }

    /// Handles completion of individual file downloads (new streaming system)
    private func handleFileDownloadCompletion(taskId: Int, fileInfo: (modelId: String, fileName: String, destination: URL), location: URL) {
        let fileKey = "\(fileInfo.modelId):\(fileInfo.fileName)"

        print("✅ File download completed: \(fileInfo.fileName) (taskId: \(taskId))")

        do {
            // Create destination directory if needed
            let directory = fileInfo.destination.deletingLastPathComponent()
            if !fileManager.fileExists(atPath: directory.path) {
                try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
            }

            // Remove existing file if present (for re-downloads)
            if fileManager.fileExists(atPath: fileInfo.destination.path) {
                try fileManager.removeItem(at: fileInfo.destination)
            }

            // Move downloaded file to destination
            try fileManager.moveItem(at: location, to: fileInfo.destination)

            // Verify file size if we have expected size
            if let expectedSize = expectedFileSizes[fileKey] {
                let attributes = try fileManager.attributesOfItem(atPath: fileInfo.destination.path)
                if let actualSize = attributes[.size] as? Int64 {
                    if actualSize < expectedSize {
                        print("⚠️ File size mismatch: \(actualSize) < \(expectedSize) for \(fileInfo.fileName)")
                    } else {
                        print("✅ File size verified: \(actualSize) bytes for \(fileInfo.fileName)")
                    }
                }
            }

            print("✅ File saved to: \(fileInfo.destination.path)")

            // Handle zip file extraction for monolithic models
            if fileInfo.fileName.hasSuffix(".mlmodelc.zip") || fileInfo.fileName.hasSuffix(".mlpackage.zip") {
                print("📦 Extracting zip file: \(fileInfo.fileName)")
                let modelDir = fileInfo.destination.deletingLastPathComponent()

                if self.extractZipFile(at: fileInfo.destination, to: modelDir) {
                    print("✅ Successfully extracted \(fileInfo.fileName) to \(modelDir.path)")

                    // Remove the zip file after successful extraction
                    do {
                        try self.fileManager.removeItem(at: fileInfo.destination)
                        print("🗑️ Removed zip file after extraction: \(fileInfo.fileName)")
                    } catch {
                        print("⚠️ Could not remove zip file after extraction: \(error)")
                    }
                } else {
                    print("❌ Failed to extract zip file \(fileInfo.fileName)")
                    // Don't fail the download - the zip file is still there for manual extraction
                }
            }

            // Defer cleanup to after delegate callback completes
            // IMPORTANT: Call completion handler on a BACKGROUND queue - the handler
            // may contain dispatch group leave() calls, and calling on main thread
            // can cause "Semaphore object deallocated while in use" crash.
            let taskIdToCleanup = taskId
            let keyToCleanup = fileKey
            DispatchQueue.global(qos: .utility).async { [weak self] in
                guard let self = self else { return }

                // Thread-safe cleanup of tracking (but NOT fileDownloadTasks)
                let completionHandler: ((Bool) -> Void)? = self.withDownloadLock {
                    self.taskToFileInfo.removeValue(forKey: taskIdToCleanup)
                    // Do NOT remove fileDownloadTasks - causes semaphore crash
                    let handler = self.fileCompletionHandlers[keyToCleanup]
                    self.fileCompletionHandlers.removeValue(forKey: keyToCleanup)
                    return handler
                }

                // Call completion handler on background queue
                completionHandler?(true)
            }

        } catch {
            print("❌ Error saving downloaded file \(fileInfo.fileName): \(error)")
            setDownloadError(for: fileInfo.modelId, message: "Failed to save \(fileInfo.fileName)")

            // Defer cleanup - call handler on BACKGROUND queue to avoid semaphore crash
            let taskIdToCleanup = taskId
            let keyToCleanup = fileKey
            DispatchQueue.global(qos: .utility).async { [weak self] in
                guard let self = self else { return }

                // Thread-safe cleanup of tracking (but NOT fileDownloadTasks)
                let completionHandler: ((Bool) -> Void)? = self.withDownloadLock {
                    self.taskToFileInfo.removeValue(forKey: taskIdToCleanup)
                    // Do NOT remove fileDownloadTasks - causes semaphore crash
                    let handler = self.fileCompletionHandlers[keyToCleanup]
                    self.fileCompletionHandlers.removeValue(forKey: keyToCleanup)
                    return handler
                }

                // Call completion handler with failure on background queue
                completionHandler?(false)
            }
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let taskId = downloadTask.taskIdentifier

        // Thread-safe access to taskToFileInfo
        let fileInfo: (modelId: String, fileName: String, destination: URL)? = withDownloadLock {
            return taskToFileInfo[taskId]
        }

        // Handle individual file download progress (new system)
        if let fileInfo = fileInfo {
            handleFileDownloadProgress(
                taskId: taskId,
                fileInfo: fileInfo,
                bytesWritten: bytesWritten,
                totalBytesWritten: totalBytesWritten,
                totalBytesExpected: totalBytesExpectedToWrite
            )
            return
        }

        // Legacy handling for top-level model downloads
        let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)

        guard let modelId = downloadTasks.first(where: {
            if let task = $0.value as? URLSessionDownloadTask {
                return task == downloadTask
            } else if let combinedTask = $0.value as? CombinedDownloadTask {
                return combinedTask.tasks.contains(downloadTask)
            }
            return false
        })?.key,
              let progressHandler = progressObservers[modelId] else {
            return
        }

        DispatchQueue.main.async {
            progressHandler(progress)
        }
    }

    /// Handles progress updates for individual file downloads with byte-based tracking
    private func handleFileDownloadProgress(taskId: Int, fileInfo: (modelId: String, fileName: String, destination: URL), bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpected: Int64) {
        let fileKey = "\(fileInfo.modelId):\(fileInfo.fileName)"
        let modelId = fileInfo.modelId

        // Store expected size for integrity validation later
        if totalBytesExpected > 0 {
            expectedFileSizes[fileKey] = totalBytesExpected
        }

        // Update bytes downloaded for this file
        bytesDownloadedPerFile[fileKey] = totalBytesWritten

        // Calculate and update total bytes for the model
        var totalDownloaded: Int64 = 0
        for (key, bytes) in bytesDownloadedPerFile {
            if key.hasPrefix("\(modelId):") {
                totalDownloaded += bytes
            }
        }
        totalBytesDownloadedForModel[modelId] = totalDownloaded

        // Update last progress time for stall detection
        lastProgressUpdateTime[modelId] = Date()

        // Progress smoothing: only use values that are >= the max we've seen
        // This prevents UI from jumping backward due to concurrent download updates
        let previousMaxBytes = maxDownloadedBytes[modelId] ?? 0
        let smoothedDownloaded = max(totalDownloaded, previousMaxBytes)
        maxDownloadedBytes[modelId] = smoothedDownloaded

        // Calculate overall progress for this model using smoothed values
        let totalExpected = totalBytesExpectedForModel[modelId] ?? totalBytesExpected
        let overallProgress = totalExpected > 0 ? Double(smoothedDownloaded) / Double(totalExpected) : 0.0

        // Also smooth the progress percentage
        let previousMaxProgress = maxDownloadProgress[modelId] ?? 0.0
        let smoothedProgress = max(overallProgress, previousMaxProgress)
        maxDownloadProgress[modelId] = smoothedProgress

        let overallProgressPercent = Int(smoothedProgress * 100)

        // Format total bytes for display (use smoothed value)
        let totalDownloadedMB = Double(smoothedDownloaded) / 1_000_000.0
        let totalExpectedMB = Double(totalExpected) / 1_000_000.0

        // Get or cache model display name
        let modelName: String
        if let cachedName = modelDisplayNames[modelId] {
            modelName = cachedName
        } else if let model = getModel(for: modelId) {
            modelName = model.name
            modelDisplayNames[modelId] = modelName
        } else {
            // Fallback: clean up modelId for display
            modelName = modelId.replacingOccurrences(of: "_", with: " ")
                .components(separatedBy: "-").last ?? modelId
            modelDisplayNames[modelId] = modelName
        }

        // Calculate ETA based on download speed
        var etaString = ""
        if let startTime = downloadStartTimes[modelId], overallProgress > 0.01 {
            let elapsedTime = Date().timeIntervalSince(startTime)
            let estimatedTotalTime = elapsedTime / overallProgress
            let remainingTime = estimatedTotalTime - elapsedTime

            if remainingTime > 0 && remainingTime < 86400 { // Less than 24 hours
                if remainingTime < 60 {
                    etaString = " - <1 min"
                } else if remainingTime < 3600 {
                    let minutes = Int(remainingTime / 60)
                    etaString = " - ~\(minutes) min"
                } else {
                    let hours = Int(remainingTime / 3600)
                    let minutes = Int((remainingTime.truncatingRemainder(dividingBy: 3600)) / 60)
                    etaString = " - ~\(hours)h \(minutes)m"
                }
            }
        }

        DispatchQueue.main.async {
            // Update progress observers with smoothed progress
            self.progressObservers[modelId]?(smoothedProgress)

            // Update status with overall progress for whole model (no individual file names)
            // Format: "ModelName: 45% (156.2/355.9 MB) - ~5 min"
            let statusMessage = String(format: "%@: %d%% (%.1f/%.1f MB)%@",
                                       modelName, overallProgressPercent,
                                       totalDownloadedMB, totalExpectedMB, etaString)
            self.currentDownloadingFiles[modelId] = statusMessage

            // Update file progress observers with smoothed progress
            self.fileProgressObservers[modelId]?(statusMessage, smoothedProgress)

            // Clear any error messages since download is progressing
            self.downloadErrorMessages.removeValue(forKey: modelId)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let taskId = task.taskIdentifier

        // Thread-safe access to taskToFileInfo
        let fileInfo: (modelId: String, fileName: String, destination: URL)? = withDownloadLock {
            return taskToFileInfo[taskId]
        }

        // Handle individual file download errors (new system)
        if let fileInfo = fileInfo {
            handleFileDownloadError(taskId: taskId, fileInfo: fileInfo, error: error)
            return
        }

        // Legacy handling for top-level model downloads
        if let error = error, !(error.localizedDescription.contains("cancelled")) {
            print("Download task failed with error: \(error)")

            // Find the model ID associated with this task
            if let modelId = downloadTasks.first(where: {
                if let downloadTask = $0.value as? URLSessionDownloadTask {
                    return downloadTask == task
                } else if let combinedTask = $0.value as? CombinedDownloadTask,
                          let downloadTask = task as? URLSessionDownloadTask {
                    return combinedTask.tasks.contains(downloadTask)
                }
                return false
            })?.key {
                DispatchQueue.main.async {
                    self.currentDownloadingFiles[modelId] = "Download failed: \(error.localizedDescription)"
                    self.modelDownloadStatus[modelId] = .failed
                    self.completionHandlers[modelId]?(false)
                }
            }
        }
    }

    /// Handles errors for individual file downloads with resume data support
    private func handleFileDownloadError(taskId: Int, fileInfo: (modelId: String, fileName: String, destination: URL), error: Error?) {
        let fileKey = "\(fileInfo.modelId):\(fileInfo.fileName)"
        let modelId = fileInfo.modelId
        let taskIdToCleanup = taskId
        let keyToCleanup = fileKey

        // Handle no error case (successful completion without didFinishDownloadingTo being called)
        // Note: didFinishDownloadingTo should always be called first for successful downloads,
        // so the completion handler should already have been called and removed from fileCompletionHandlers.
        guard let error = error else {
            print("✅ File download task completed without error: \(fileInfo.fileName) (didCompleteWithError path)")

            // Check if completion handler still exists (shouldn't if didFinishDownloadingTo was called)
            let handlerStillExists = self.withDownloadLock {
                return self.fileCompletionHandlers[keyToCleanup] != nil
            }

            if handlerStillExists {
                print("⚠️ WARNING: Completion handler still exists for \(fileInfo.fileName) - didFinishDownloadingTo may not have been called!")
                // Call the handler to prevent hanging
                DispatchQueue.global(qos: .utility).async { [weak self] in
                    guard let self = self else { return }
                    let handler: ((Bool) -> Void)? = self.withDownloadLock {
                        self.taskToFileInfo.removeValue(forKey: taskIdToCleanup)
                        let h = self.fileCompletionHandlers[keyToCleanup]
                        self.fileCompletionHandlers.removeValue(forKey: keyToCleanup)
                        return h
                    }
                    handler?(true)
                }
            } else {
                // Normal case - just cleanup
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.withDownloadLock {
                        _ = self.taskToFileInfo.removeValue(forKey: taskIdToCleanup)
                    }
                }
            }
            return
        }

        // Check if this was a cancellation
        let nsError = error as NSError
        if nsError.code == NSURLErrorCancelled {
            print("[DOWNLOAD] ⏸️ CANCELLED: \(fileInfo.fileName)")

            // Try to extract resume data for later
            var resumeDataToStore: Data? = nil
            if let resumeData = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
                print("💾 Saved resume data for: \(fileInfo.fileName) (\(resumeData.count) bytes)")
                resumeDataToStore = resumeData
            }

            // Defer cleanup - use background queue for handler to avoid semaphore crash
            DispatchQueue.global(qos: .utility).async { [weak self] in
                guard let self = self else { return }

                // Store resume data if available
                if let resumeData = resumeDataToStore {
                    self.withDownloadLock {
                        self.resumeDataStore[keyToCleanup] = resumeData
                    }
                }

                // Cleanup tracking (but NOT fileDownloadTasks)
                let handler: ((Bool) -> Void)? = self.withDownloadLock {
                    self.taskToFileInfo.removeValue(forKey: taskIdToCleanup)
                    // Do NOT remove fileDownloadTasks - causes semaphore crash
                    let h = self.fileCompletionHandlers[keyToCleanup]
                    self.fileCompletionHandlers.removeValue(forKey: keyToCleanup)
                    return h
                }
                handler?(false)
            }
            return
        }

        // Handle actual errors - check if we should auto-retry
        let currentRetryCount = fileRetryCount[keyToCleanup] ?? 0
        let canRetry = currentRetryCount < maxRetryAttempts

        logDownload("❌ File failed: \(fileInfo.fileName) - \(error.localizedDescription) (attempt \(currentRetryCount + 1)/\(maxRetryAttempts + 1))")

        // Extract resume data if available (for network errors)
        var resumeDataToStore: Data? = nil
        if let resumeData = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
            logDownload("💾 Saved resume data: \(fileInfo.fileName) (\(resumeData.count) bytes)")
            resumeDataToStore = resumeData
        }

        // Store resume data immediately (thread-safe)
        if let resumeData = resumeDataToStore {
            self.withDownloadLock {
                self.resumeDataStore[keyToCleanup] = resumeData
            }
        }

        // Auto-retry for retryable errors
        let isRetryable = [NSURLErrorTimedOut, NSURLErrorNetworkConnectionLost,
                          NSURLErrorCannotConnectToHost, NSURLErrorNotConnectedToInternet].contains(nsError.code)

        if canRetry && (isRetryable || resumeDataToStore != nil) {
            // Increment retry count
            fileRetryCount[keyToCleanup] = currentRetryCount + 1

            logDownload("🔄 AUTO-RETRY: Retrying \(fileInfo.fileName) in 5 seconds (attempt \(currentRetryCount + 2)/\(maxRetryAttempts + 1))")

            // Update UI to show retry
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.currentDownloadingFiles[modelId] = "Retrying \(fileInfo.fileName)... (attempt \(currentRetryCount + 2))"
            }

            // Retry after delay
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 5.0) { [weak self] in
                guard let self = self else { return }

                // Get the handler before retrying
                let handler: ((Bool) -> Void)? = self.withDownloadLock {
                    return self.fileCompletionHandlers[keyToCleanup]
                }

                // Retry the download
                self.downloadSingleFileWithRetry(
                    modelId: modelId,
                    fileName: fileInfo.fileName,
                    destination: fileInfo.destination,
                    completion: handler
                )
            }
            return
        }

        // Create user-friendly error message (no more retries)
        var userMessage: String
        switch nsError.code {
        case NSURLErrorTimedOut:
            userMessage = "Connection timed out for \(fileInfo.fileName) after \(maxRetryAttempts + 1) attempts."
        case NSURLErrorNetworkConnectionLost:
            userMessage = "Network connection lost while downloading \(fileInfo.fileName)."
        case NSURLErrorNotConnectedToInternet:
            userMessage = "No internet connection. Please check your network settings."
        case NSURLErrorCannotFindHost, NSURLErrorCannotConnectToHost:
            userMessage = "Cannot reach download server after \(maxRetryAttempts + 1) attempts."
        default:
            userMessage = "Download failed: \(error.localizedDescription)"
        }

        logDownload("❌ GAVE UP: \(fileInfo.fileName) after \(currentRetryCount + 1) attempts")

        // Update UI on main thread (no completion handler here)
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.downloadErrorMessages[modelId] = userMessage
            self.currentDownloadingFiles[modelId] = "Error: \(userMessage)"
            self.modelDownloadStatus[modelId] = .failed
        }

        // Cleanup and call handler on background queue to avoid semaphore crash
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self = self else { return }

            // Cleanup retry count
            self.fileRetryCount.removeValue(forKey: keyToCleanup)

            // Cleanup tracking (but NOT fileDownloadTasks)
            let handler: ((Bool) -> Void)? = self.withDownloadLock {
                self.taskToFileInfo.removeValue(forKey: taskIdToCleanup)
                // Do NOT remove fileDownloadTasks - causes semaphore crash
                let h = self.fileCompletionHandlers[keyToCleanup]
                self.fileCompletionHandlers.removeValue(forKey: keyToCleanup)
                return h
            }
            handler?(false)
        }
    }

    // MARK: - Zip Extraction

    /// Extracts a zip file to the specified destination directory
    /// Uses posix_spawn to call /usr/bin/unzip which works on macOS, Mac Catalyst, and iOS
    private func extractZipFile(at zipURL: URL, to destinationURL: URL) -> Bool {
        #if os(macOS) || targetEnvironment(macCatalyst)
        // Use posix_spawn to call unzip - works on macOS and Mac Catalyst
        return extractZipUsingPosixSpawn(at: zipURL, to: destinationURL)
        #else
        // For iOS devices, zip extraction requires bundled binaries or third-party library
        print("⚠️ Zip extraction not supported on iOS devices")
        print("📂 Please manually extract the zip file: \(zipURL.path)")
        return false
        #endif
    }

    #if os(macOS) || targetEnvironment(macCatalyst)
    /// Uses posix_spawn to call /usr/bin/unzip - works on Mac Catalyst unlike Process
    private func extractZipUsingPosixSpawn(at zipURL: URL, to destinationURL: URL) -> Bool {
        print("📦 Extracting zip using posix_spawn...")
        print("   Source: \(zipURL.path)")
        print("   Destination: \(destinationURL.path)")

        // Prepare arguments for unzip: /usr/bin/unzip -o -q zipfile -d destdir
        let unzipPath = "/usr/bin/unzip"
        let args = [unzipPath, "-o", "-q", zipURL.path, "-d", destinationURL.path]

        // Convert to C strings
        var cArgs = args.map { strdup($0) }
        cArgs.append(nil)

        defer {
            // Free the duplicated strings
            for i in 0..<(cArgs.count - 1) {
                free(cArgs[i])
            }
        }

        var pid: pid_t = 0
        var fileActions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&fileActions)

        // Redirect stdout and stderr to /dev/null for quiet operation
        posix_spawn_file_actions_addopen(&fileActions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0)
        posix_spawn_file_actions_addopen(&fileActions, STDERR_FILENO, "/dev/null", O_WRONLY, 0)

        let status = posix_spawn(&pid, unzipPath, &fileActions, nil, &cArgs, nil)
        posix_spawn_file_actions_destroy(&fileActions)

        if status != 0 {
            print("❌ posix_spawn failed with status \(status)")
            return false
        }

        // Wait for the unzip process to complete
        var exitStatus: Int32 = 0
        waitpid(pid, &exitStatus, 0)

        // Check if process exited normally and with success code
        // WIFEXITED(status) = ((status) & 0x7f) == 0
        // WEXITSTATUS(status) = ((status) >> 8) & 0xff
        let normalExit = (exitStatus & 0x7f) == 0
        if normalExit {
            let exitCode = (exitStatus >> 8) & 0xff
            if exitCode == 0 {
                print("✅ unzip completed successfully")
                return true
            } else {
                print("❌ unzip failed with exit code \(exitCode)")
                return false
            }
        } else {
            print("❌ unzip process terminated abnormally")
            return false
        }
    }
    #endif

    // MARK: - Helper Methods

    /// Loads custom models from storage
    private func loadCustomModels() {
        // Only load from models.json, removing UserDefaults backward compatibility
        loadModelsFromJSON()
        
        // No longer load from UserDefaults
        print("Using models.json as the sole source for custom models")
    }
    
    /// Loads the previously selected model from UserDefaults
    public func loadSelectedModel() {
        print("🔄 Loading previously selected model from UserDefaults")
        
        if let savedModelId = UserDefaults.standard.string(forKey: selectedModelIdKey) {
            print("📱 Found saved model ID in UserDefaults: \(savedModelId)")
            
            // First try to find the model directly without using getModel
            if let model = availableModels.first(where: { $0.id == savedModelId }) {
                selectedModel = model
                print("✅ Loaded previously selected model: \(model.name)")
                
                // Verify the model directory exists directly
                let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
                let modelsDirectory = documentsDirectory.appendingPathComponent("Models")
                
                let modelPath: URL
                if model.id == "llama-3.2-1b" {
                    modelPath = modelsDirectory.appendingPathComponent("llama_3_2_1b_iosv2_0")
                } else {
                    let sanitizedId = sanitizeModelId(model.id)
                    modelPath = modelsDirectory.appendingPathComponent(sanitizedId)
                }
                
                if !fileManager.fileExists(atPath: modelPath.path) {
                    print("⚠️ Selected model directory does not exist at \(modelPath.path)")
                    model.isDownloaded = false
                }
                
                return
            }
            
            // Special case for default model
            if savedModelId.contains("llama-3.2") || 
               savedModelId.contains("anemll-llama-3.2") || 
               savedModelId.contains("anemll/anemll-llama-3.2") {
                
                if let defaultModel = availableModels.first(where: { $0.id == "llama-3.2-1b" }) {
                    selectedModel = defaultModel
                    print("✅ Mapped saved ID \(savedModelId) to default model: \(defaultModel.id)")
                    
                    // Update UserDefaults with the correct ID
                    UserDefaults.standard.set(defaultModel.id, forKey: selectedModelIdKey)
                    return
                }
            }
            
            // Try with sanitized version matching
            let sanitizedSavedId = sanitizeModelId(savedModelId)
            if let model = availableModels.first(where: { sanitizeModelId($0.id) == sanitizedSavedId }) {
                selectedModel = model
                print("✅ Matched using sanitized ID: \(savedModelId) -> \(model.id)")
                
                // Update UserDefaults with the correct ID
                UserDefaults.standard.set(model.id, forKey: selectedModelIdKey)
                return
            }
            
            print("⚠️ Could not find model matching saved ID: \(savedModelId)")
        } else {
            print("📱 No previously selected model found in UserDefaults")
        }
        
        // If no model was previously selected or we couldn't find it,
        // select the first downloaded model or the default model
        if let firstDownloadedModel = availableModels.first(where: { $0.isDownloaded }) {
            selectedModel = firstDownloadedModel
            print("🔄 Auto-selected first downloaded model: \(firstDownloadedModel.name)")
        } else if let defaultModel = availableModels.first(where: { $0.id == "llama-3.2-1b" }) {
            selectedModel = defaultModel
            print("🔄 Auto-selected default model: \(defaultModel.name)")
        }
    }
    
    /// Formats a file size into a human-readable string
    private func formatFileSize(_ size: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(size))
    }
    
    /// Cleans up download resources for a model ID
    private func cleanupDownload(for modelId: String) {
        print("DEBUG: Cleaning up download resources for model: \(modelId)")
        
        // Remove download task
        downloadTasks.removeValue(forKey: modelId)
        
        // Remove observers
        progressObservers.removeValue(forKey: modelId)
        fileProgressObservers.removeValue(forKey: modelId)
        downloadProgressObservers.removeValue(forKey: modelId)
        
        // Clear completion handler
        completionHandlers.removeValue(forKey: modelId)
        
        // Clear tracking information
        currentDownloadingFiles.removeValue(forKey: modelId)
        downloadProgress.removeValue(forKey: modelId)
        lastProgressUpdate.removeValue(forKey: modelId)
        
        // Stop activity indicator
        stopDownloadActivityIndicator(for: modelId)
        
        // Allow device to sleep after download completes
        allowSleepAfterDownload()
        
        // Perform a verification to update model status correctly
        if let model = getModel(for: modelId) {
            DispatchQueue.main.async {
                model.refreshDownloadStatus()
                
                // Post notification about download status change
                NotificationCenter.default.post(
                    name: Notification.Name("ModelDownloadStatusChanged"),
                    object: nil, 
                    userInfo: ["modelId": modelId]
                )
            }
        }
    }
    
    // A comprehensive method to get required files based on model configuration
    func getRequiredFiles(from config: ModelConfiguration) -> [String] {
        // Paths for files (not directories)
        var requiredFiles = [
            "meta.yaml",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]

        // Directory names (.mlmodelc directories)
        var requiredDirectories = [String]()

        // Detailed logging only when VERBOSE_MODEL_VERIFICATION environment is set
        // This reduces log noise during normal operation
        #if DEBUG
        // Minimal logging for debugging - only show model type
        // print("🔍 Model: \(config.modelPrefix), chunks=\(config.numChunks), monolithic=\(config.isMonolithic)")
        #endif

        // Handle monolithic models differently
        if config.isMonolithic, let monolithicModelName = config.monolithicModel {
            print("🎯 MONOLITHIC MODEL DETECTED: \(monolithicModelName)")

            // Monolithic models can be distributed either as:
            // 1. A zip file (xxx.mlmodelc.zip) - will be extracted after download
            // 2. A directory (xxx.mlmodelc) - download weights/weight.bin directly
            // We add BOTH options - download/verification will use whichever is available

            // Option 1: Add the monolithic model directory (will download weights/weight.bin)
            requiredDirectories.append(monolithicModelName)
            print("📁 Added monolithic model directory: \(monolithicModelName)")

            // Option 2: Also add the zip file as an alternative (for models distributed as zip)
            let zipFileName = "\(monolithicModelName).zip"
            requiredFiles.append(zipFileName)
            print("📦 Added monolithic model zip (alternative): \(zipFileName)")

            // Combine directories and files
            let allRequired = requiredDirectories + requiredFiles

            print("Total required for monolithic model: \(allRequired.count) files")
            return allRequired
        }

        // Standard chunked model handling below
        
        // 1. Embeddings model directory structure
        let embedDirName: String
        if let lutEmbeddings = config.lutEmbeddings, lutEmbeddings > 0 {
            embedDirName = "\(config.modelPrefix)_embeddings_lut\(lutEmbeddings).mlmodelc"
        } else {
            embedDirName = "\(config.modelPrefix)_embeddings.mlmodelc"
        }
        
        // Add embedding directory
        requiredDirectories.append(embedDirName)
        
        // Add files within embedding directory (excluding model.mlmodel)
        requiredFiles.append("\(embedDirName)/model.mil")
        requiredFiles.append("\(embedDirName)/metadata.json")
        requiredFiles.append("\(embedDirName)/analytics/coremldata.bin")
        requiredFiles.append("\(embedDirName)/weights/weight.bin")
        requiredFiles.append("\(embedDirName)/coremldata.bin") // Add root coremldata.bin file
        
        // 2. LM head model directory structure
        let lmHeadDirName: String
        if let lutLMHead = config.lutLMHead, lutLMHead > 0 {
            lmHeadDirName = "\(config.modelPrefix)_lm_head_lut\(lutLMHead).mlmodelc"
        } else {
            // If LUT LM head is nil or 0, use the version without LUT suffix
            lmHeadDirName = "\(config.modelPrefix)_lm_head.mlmodelc"
        }
        
        // Add lmhead directory
        requiredDirectories.append(lmHeadDirName)
        
        // Add files within lmhead directory (excluding model.mlmodel)
        requiredFiles.append("\(lmHeadDirName)/model.mil")
        requiredFiles.append("\(lmHeadDirName)/metadata.json")
        requiredFiles.append("\(lmHeadDirName)/analytics/coremldata.bin")
        requiredFiles.append("\(lmHeadDirName)/weights/weight.bin")
        requiredFiles.append("\(lmHeadDirName)/coremldata.bin") // Add root coremldata.bin file
        
        // 3. FFN chunks directory structures - using config.numChunks
        // If lutFFN is specified and > 0, use that value
        if let lutFFN = config.lutFFN, lutFFN > 0 {
            for i in 1...config.numChunks {
                let chunkDirName = String(format: "\(config.modelPrefix)_FFN_PF_lut\(lutFFN)_chunk_%02dof%02d.mlmodelc",
                                         i, config.numChunks)

                // Add chunk directory
                requiredDirectories.append(chunkDirName)
                
                // Add files within chunk directory (excluding model.mlmodel)
                requiredFiles.append("\(chunkDirName)/model.mil")
                requiredFiles.append("\(chunkDirName)/metadata.json")
                requiredFiles.append("\(chunkDirName)/analytics/coremldata.bin")
                requiredFiles.append("\(chunkDirName)/weights/weight.bin")
                requiredFiles.append("\(chunkDirName)/coremldata.bin") // Add root coremldata.bin file
            }
        } else {
            // If LUT FFN is nil or 0, use without LUT suffix
            for i in 1...config.numChunks {
                let chunkDirName = String(format: "\(config.modelPrefix)_FFN_PF_chunk_%02dof%02d.mlmodelc",
                                         i, config.numChunks)

                // Add chunk directory
                requiredDirectories.append(chunkDirName)
                
                // Add files within chunk directory (excluding model.mlmodel)
                requiredFiles.append("\(chunkDirName)/model.mil")
                requiredFiles.append("\(chunkDirName)/metadata.json")
                requiredFiles.append("\(chunkDirName)/analytics/coremldata.bin")
                requiredFiles.append("\(chunkDirName)/weights/weight.bin")
                requiredFiles.append("\(chunkDirName)/coremldata.bin") // Add root coremldata.bin file
            }
        }
        
        // Combine directories and files, with directories first
        let allRequired = requiredDirectories + requiredFiles
        return allRequired
    }
    
    /// Selects a model and updates the selectedModel property
    @MainActor
    public func selectModel(_ model: Model) {
        // Check if model is downloaded before selecting
        if !model.isDownloaded {
            print("⚠️ Warning: Attempting to select a model that is not downloaded: \(model.name)")
            // We'll still select it but log a warning
        }
        
        // Use exact ID check for the default model
        let modelIdToStore: String
        if model.id == "llama-3.2-1b" {
            // For the default model, always use the consistent ID
            modelIdToStore = "llama-3.2-1b"
            print("🔄 Using standardized ID for default model: \(modelIdToStore)")
        } else {
            modelIdToStore = model.id
        }
        
        // Update state immediately without delay
        self.selectedModel = model
        UserDefaults.standard.set(modelIdToStore, forKey: self.selectedModelIdKey)
        print("✅ Selected model: \(model.name) with ID: \(modelIdToStore)")
    }
    
    /// Checks if a model is currently selected
    func isModelSelected(_ model: Model) -> Bool {
        return selectedModel?.id == model.id
    }
    
    /// Gets the currently selected model
    func getSelectedModel() -> Model? {
        return selectedModel
    }
    
    /// Gets the currently downloading file for a model
    func getCurrentDownloadingFile(for modelId: String) -> String? {
        return currentDownloadingFiles[modelId]
    }
    
    /// Gets the download progress for a model
    func getDownloadProgress(for modelId: String) -> Double? {
        return downloadProgress[modelId]
    }
    
    /// Checks if the app has write permission to the model directory
    func hasWritePermission() -> Bool {
        let testFilePath = modelStorageDirectory.appendingPathComponent("write_test.txt")
        do {
            try "Test write access".write(to: testFilePath, atomically: true, encoding: .utf8)
            try fileManager.removeItem(at: testFilePath)
            return true
        } catch {
            print("No write permission: \(error)")
            return false
        }
    }
    
    /// Loads a model for inference
    func loadModelForInference(_ model: Model) async throws {
        // Implementation would depend on your inference setup
        print("Loading model for inference: \(model.name)")
        // This is a placeholder - you would implement the actual loading logic
    }
    
    /// Removes a custom model
    func removeCustomModel(modelId: String) {
        // Remove from available models
        if let index = availableModels.firstIndex(where: { $0.id == modelId }) {
            let model = availableModels[index]
            availableModels.remove(at: index)
            
            // Remove from downloaded models if needed
            if let downloadedIndex = downloadedModels.firstIndex(where: { $0.id == modelId }) {
                downloadedModels.remove(at: downloadedIndex)
            }
            
            // Remove from disk if downloaded
            if model.isDownloaded {
                let modelPath = modelStorageDirectory.appendingPathComponent(modelId)
                do {
                    try fileManager.removeItem(at: modelPath)
                    print("Removed model files for: \(modelId)")
                } catch {
                    print("Error removing model files: \(error)")
                }
            }
            
            // Update UserDefaults
            updateCustomModelsInUserDefaults()
        }
    }
    
    /// Adds a custom model
    func addCustomModel(
        name: String?,
        description: String?,
        id: String? = nil,
        size: Int = 0,
        downloadURL: String = "",
        completion: ((Bool, String?) -> Void)? = nil
    ) {
        // Mark that we're adding a custom model to prevent recursion
        let isAlreadyAddingModel = ModelService.isAddingCustomModel
        ModelService.isAddingCustomModel = true
        defer { 
            // Only reset the flag if we were the ones who set it
            if !isAlreadyAddingModel {
                ModelService.isAddingCustomModel = false
            }
        }
        
        // If we have a download URL, use our improved method to avoid duplicates
        if !downloadURL.isEmpty {
            print("Using improved addCustomModel method with deduplication for URL: \(downloadURL)")
            
            // Call our improved method that handles deduplication
            addCustomModel(name: name, description: description, downloadURL: downloadURL) { success, message in
                completion?(success, message)
            }
            return
        }
        
        // For models without a download URL, use the original logic
        let modelName = name ?? "Custom Model"
        let modelDescription = description ?? "Custom model added by user"
        
        print("Adding custom model without download URL: \(modelName)")
        
        // Generate model ID based on URL if possible
        let modelId = id ?? UUID().uuidString
        
        // Create the model object
        let newModel = Model(
            id: modelId,
            name: modelName,
            description: modelDescription,
            size: size,
            downloadURL: downloadURL
        )
        
        // Add to available models
        availableModels.append(newModel)
        
        // Save to UserDefaults
        updateCustomModelsInUserDefaults()
        
        // Call completion handler
        completion?(true, nil)
    }
    
    /// Gets a user-friendly path for display
    func getUserFriendlyPath() -> String {
        // Convert the full path to a more user-friendly format
        #if os(macOS) || targetEnvironment(macCatalyst)
        return "Documents/Models"
        #else
        return "Documents/Models"
        #endif
    }
    
    /// Gets a user-friendly display URL for a model
    func getDisplayURL(for model: Model) -> String {
        // If it's a Hugging Face model, format it nicely
        if model.downloadURL.hasPrefix("https://huggingface.co/") {
            // Extract the repo path from the URL
            let prefix = "https://huggingface.co/"
            let repoPath = model.downloadURL.dropFirst(prefix.count)
            return "🤗 Hugging Face: \(repoPath)"
        } else if model.downloadURL.hasPrefix("huggingface://") {
            // Extract the repo path from the special URL format
            let prefix = "huggingface://"
            let repoPath = model.downloadURL.dropFirst(prefix.count)
            return "🤗 Hugging Face: \(repoPath)"
        } else if model.downloadURL.isEmpty {
            return "Local model"
        } else {
            // For other URLs, just return as is but truncate if too long
            let maxLength = 40
            if model.downloadURL.count > maxLength {
                let startIndex = model.downloadURL.startIndex
                let endIndex = model.downloadURL.index(startIndex, offsetBy: maxLength)
                return "\(model.downloadURL[startIndex..<endIndex])..."
            }
            return model.downloadURL
        }
    }
    
    // Method to get a model by ID
    public func getModel(for modelId: String) -> Model? {
        // First try exact match
        if let model = availableModels.first(where: { $0.id == modelId }) {
            return model
        }

        // Special case for default model
        if modelId == "llama-3.2-1b" ||
           modelId == "llama_3_2_1b_iosv2_0" ||
           modelId.contains("anemll-llama-3.2") ||
           modelId.contains("anemll/anemll-llama-3.2") {

            if let defaultModel = availableModels.first(where: { $0.id == "llama-3.2-1b" }) {
                return defaultModel
            }
        }

        // If not found, try case-insensitive match
        if let model = availableModels.first(where: { $0.id.lowercased() == modelId.lowercased() }) {
            return model
        }

        // Try with sanitized version of the ID
        let sanitizedId = sanitizeModelId(modelId)
        if let model = availableModels.first(where: { sanitizeModelId($0.id) == sanitizedId }) {
            return model
        }

        print("⚠️ Model not found: \(modelId)")
        return nil
    }
    
    /// Verifies that all required files for a model exist
    public func verifyModelFiles(modelId: String, verbose: Bool = true) -> Bool {
        let modelDir = getModelPath(for: modelId)
        
        guard fileManager.fileExists(atPath: modelDir.path) else {
            print("Model directory doesn't exist: \(modelDir.path)")
            return false
        }
        
        // Get model size for comparison
        let model = getModel(for: modelId)
        let expectedSize = model?.size ?? 0
        var actualTotalSize: Int64 = 0
        var fileSizes = [String: Int64]()
        
        let metaYamlPath = modelDir.appendingPathComponent("meta.yaml")
        let configJsonPath = modelDir.appendingPathComponent("config.json")
        let tokenizerJsonPath = modelDir.appendingPathComponent("tokenizer.json")
        let tokenizerConfigJsonPath = modelDir.appendingPathComponent("tokenizer_config.json")
        
        let metaYamlExists = fileManager.fileExists(atPath: metaYamlPath.path)
        let configJsonExists = fileManager.fileExists(atPath: configJsonPath.path)
        let tokenizerJsonExists = fileManager.fileExists(atPath: tokenizerJsonPath.path)
        let tokenizerConfigJsonExists = fileManager.fileExists(atPath: tokenizerConfigJsonPath.path)
        
        if verbose {
            print("📋 Verification for model: \(modelId)")
            print("Basic files:")
            print("  - meta.yaml: \(metaYamlExists ? "✅" : "❌")")
            print("  - config.json: \(configJsonExists ? "✅" : "❌")")
            print("  - tokenizer.json: \(tokenizerJsonExists ? "✅" : "❌")")
            print("  - tokenizer_config.json: \(tokenizerConfigJsonExists ? "✅" : "❌")")
        }
        
        var allRequiredFilesExist = true
        var missingFiles: [String] = []
        var missingWeightFiles: [String] = []
        var hasMissingCriticalFiles = false
        
        // Check for critical files first
        if !metaYamlExists {
            missingFiles.append("meta.yaml")
            hasMissingCriticalFiles = true
        }
        if !configJsonExists {
            missingFiles.append("config.json")
            hasMissingCriticalFiles = true
        }
        if !tokenizerJsonExists {
            missingFiles.append("tokenizer.json")
            hasMissingCriticalFiles = true
        }
        if !tokenizerConfigJsonExists {
            missingFiles.append("tokenizer_config.json")
            hasMissingCriticalFiles = true
        }
        
        // Update model's hasPlaceholders status based on critical files
        if let model = model {
            if hasMissingCriticalFiles {
                print("⚠️ Model \(modelId) has missing critical files - setting hasPlaceholders to true")
                DispatchQueue.main.async {
                    model.hasPlaceholders = true
                }
            } else {
                DispatchQueue.main.async {
                    model.hasPlaceholders = false
                }
            }
        }
        
        // If any critical files are missing, return false immediately
        if hasMissingCriticalFiles {
            if verbose {
                print("❌ Critical files missing:")
                for file in missingFiles {
                    print("   - \(file)")
                }
            }
            return false
        }
        
        // Continue with weight file verification
        // ... rest of the existing verification code ...
        
        // Helper function to get file size
        func getFileSize(at path: String) -> Int64 {
            do {
                let attributes = try fileManager.attributesOfItem(atPath: path)
                if let size = attributes[.size] as? Int64 {
                    return size
                }
            } catch {
                print("Error getting file size at \(path): \(error)")
            }
            return 0
        }
        
        // Format file size to human-readable
        func formatFileSize(_ size: Int64) -> String {
            let formatter = ByteCountFormatter()
            formatter.allowedUnits = [.useGB, .useMB, .useKB]
            formatter.countStyle = .file
            return formatter.string(fromByteCount: size)
        }
        
        // More thorough check if meta.yaml exists - use the exact configuration specified
        if metaYamlExists {
            do {
                // Read and parse meta.yaml to get the exact configuration
                let yamlContent = try String(contentsOf: metaYamlPath, encoding: .utf8)
                print("Successfully read meta.yaml")
                
                // Attempt to parse the ModelConfiguration to get precise values for verification
                let modelConfig = try ModelConfiguration(from: yamlContent, modelPath: modelDir.path)
                print("Using exact configuration from meta.yaml for verification:")
                print("  - Model prefix: \(modelConfig.modelPrefix)")
                print("  - Num chunks: \(modelConfig.numChunks)")
                if let lutLMHead = modelConfig.lutLMHead {
                    print("  - LUT LM Head: \(lutLMHead)")
                } else {
                    print("  - LUT LM Head: Not specified")
                }
                if let lutFFN = modelConfig.lutFFN {
                    print("  - LUT FFN: \(lutFFN)")
                } else {
                    print("  - LUT FFN: Not specified")
                }
                
                // Get the required files based on the configuration
                let requiredFiles = getRequiredFiles(from: modelConfig)
                
                print("\n🔍 Checking for \(requiredFiles.count) required files...")
                
                // Minimum size for weight.bin files (1KB) - anything smaller is likely corrupt/placeholder
                let minimumWeightFileSize: Int64 = 1024
                var emptyWeightFiles: [String] = []

                // Track if we found valid monolithic weights (via either zip extraction or directory)
                var foundValidMonolithicWeights = false
                var monolithicModelName: String? = nil
                if modelConfig.isMonolithic {
                    monolithicModelName = modelConfig.monolithicModel
                }

                // Now check all required files
                for file in requiredFiles {
                    // Special handling for monolithic model zip files
                    // After extraction, the zip is deleted and we should check for the extracted directory
                    var fileToCheck = file
                    var isMonolithicZip = false
                    var isMonolithicDir = false

                    // Check if this is the monolithic model directory entry
                    if let monoName = monolithicModelName, file == monoName {
                        isMonolithicDir = true
                        let dirPath = modelDir.appendingPathComponent(file)

                        // Check if the directory exists with weight file inside
                        if fileManager.fileExists(atPath: dirPath.path) {
                            let weightFilePath = dirPath.appendingPathComponent("weights/weight.bin")
                            if fileManager.fileExists(atPath: weightFilePath.path) {
                                let weightSize = getFileSize(at: weightFilePath.path)
                                if weightSize >= minimumWeightFileSize {
                                    // Good! Monolithic model directory is valid
                                    print("  ✅ \(file) (directory, weight: \(formatFileSize(weightSize)))")
                                    actualTotalSize += weightSize
                                    foundValidMonolithicWeights = true

                                    // Also add other files in the directory to total size
                                    if let enumerator = fileManager.enumerator(atPath: dirPath.path) {
                                        while let subFile = enumerator.nextObject() as? String {
                                            if subFile != "weights/weight.bin" {
                                                let subFilePath = dirPath.appendingPathComponent(subFile).path
                                                var isDir: ObjCBool = false
                                                if fileManager.fileExists(atPath: subFilePath, isDirectory: &isDir) && !isDir.boolValue {
                                                    actualTotalSize += getFileSize(at: subFilePath)
                                                }
                                            }
                                        }
                                    }
                                    continue  // Skip to next file - this one is valid
                                }
                            }
                        }
                        // If directory doesn't exist or is invalid, we might have the zip instead
                        // Don't fail yet - check if we already found valid weights via zip
                        if foundValidMonolithicWeights {
                            continue  // Already have valid weights, skip this
                        }
                        // Fall through to normal file checking (will mark as missing)
                    }

                    if file.hasSuffix(".mlmodelc.zip") || file.hasSuffix(".mlpackage.zip") {
                        isMonolithicZip = true
                        // Remove .zip extension to get the directory name
                        let extractedDirName = String(file.dropLast(4)) // Remove ".zip"
                        let extractedDirPath = modelDir.appendingPathComponent(extractedDirName)

                        // Check if the extracted directory exists
                        if fileManager.fileExists(atPath: extractedDirPath.path) {
                            // Extracted directory exists - check for weight file inside
                            let weightFilePath = extractedDirPath.appendingPathComponent("weights/weight.bin")
                            if fileManager.fileExists(atPath: weightFilePath.path) {
                                let weightSize = getFileSize(at: weightFilePath.path)
                                if weightSize >= minimumWeightFileSize {
                                    // Good! Monolithic model is properly extracted
                                    print("  ✅ \(extractedDirName) (extracted, weight: \(formatFileSize(weightSize)))")
                                    actualTotalSize += weightSize
                                    foundValidMonolithicWeights = true

                                    // Also add other files in the directory to total size
                                    if let enumerator = fileManager.enumerator(atPath: extractedDirPath.path) {
                                        while let subFile = enumerator.nextObject() as? String {
                                            if subFile != "weights/weight.bin" {
                                                let subFilePath = extractedDirPath.appendingPathComponent(subFile).path
                                                var isDir: ObjCBool = false
                                                if fileManager.fileExists(atPath: subFilePath, isDirectory: &isDir) && !isDir.boolValue {
                                                    actualTotalSize += getFileSize(at: subFilePath)
                                                }
                                            }
                                        }
                                    }
                                    continue  // Skip to next file - this one is valid
                                } else {
                                    print("  ❌ \(extractedDirName) (extracted but weight file empty/corrupt)")
                                    emptyWeightFiles.append("\(extractedDirName)/weights/weight.bin")
                                    missingWeightFiles.append("\(extractedDirName)/weights/weight.bin")
                                    allRequiredFilesExist = false
                                    continue
                                }
                            } else {
                                print("  ❌ \(extractedDirName) (extracted but missing weight file)")
                                missingWeightFiles.append("\(extractedDirName)/weights/weight.bin")
                                allRequiredFilesExist = false
                                continue
                            }
                        }
                        // If extracted dir doesn't exist, check if we have valid weights from directory
                        if foundValidMonolithicWeights {
                            print("  ⏭️ \(file) (skipped - using directory weights)")
                            continue  // Already have valid weights via directory, skip zip
                        }
                        // If extracted dir doesn't exist, fall through to check for the zip file
                    }

                    let filePath = modelDir.appendingPathComponent(fileToCheck)
                    let exists = fileManager.fileExists(atPath: filePath.path)

                    // Get file size for existing files
                    var fileSize: Int64 = 0
                    if exists {
                        fileSize = getFileSize(at: filePath.path)
                        actualTotalSize += fileSize
                        fileSizes[file] = fileSize
                    }

                    // Check if this is a weight file with 0 or very small size (corrupt/placeholder)
                    let isWeightFile = file.contains("/weights/weight.bin")
                    // Monolithic zip files should also be checked for minimum size (at least 100KB for a real zip)
                    let minimumZipFileSize: Int64 = 100_000  // 100KB minimum for zip files
                    let isEmptyOrCorrupt = exists && (
                        (isWeightFile && fileSize < minimumWeightFileSize) ||
                        (isMonolithicZip && fileSize < minimumZipFileSize)
                    )

                    // Critical files that must exist for the model to work
                    let isCriticalFile = file == "meta.yaml" ||
                                         file == "config.json" ||
                                         file == "tokenizer.json" ||
                                         file == "tokenizer_config.json" ||  // Add tokenizer_config.json as critical
                                         isWeightFile ||
                                         isMonolithicZip  // Monolithic zip files are critical

                    if !exists {
                        missingFiles.append(file)
                        // Monolithic zip files are essentially weight files - they contain the model weights
                        if isWeightFile || isMonolithicZip {
                            missingWeightFiles.append(file)
                            if isMonolithicZip {
                                print("❌ MONOLITHIC MODEL: Zip file not found and not extracted: \(file)")
                                print("   💡 Please download the model to get the monolithic model weights")
                            }
                        }

                        if isCriticalFile {
                            allRequiredFilesExist = false
                            if verbose {
                                print("❌ Critical file missing: \(file)")
                            }
                        }
                    } else if isEmptyOrCorrupt {
                        // File exists but is empty or corrupt - treat as missing
                        emptyWeightFiles.append(file)
                        missingWeightFiles.append(file)
                        allRequiredFilesExist = false
                        if verbose {
                            if isMonolithicZip {
                                print("❌ MONOLITHIC MODEL: Zip file is empty/corrupt (\(formatFileSize(fileSize))): \(file)")
                                print("   💡 Please re-download the model to get the monolithic model weights")
                            } else {
                                print("❌ Weight file is empty/corrupt (\(formatFileSize(fileSize))): \(file)")
                            }
                        }
                    }

                    // Print status with proper emoji and size information
                    let statusEmoji: String
                    if isEmptyOrCorrupt {
                        statusEmoji = "❌"  // Empty/corrupt file
                    } else if exists {
                        statusEmoji = "✅"  // File exists with valid size
                    } else if isCriticalFile {
                        statusEmoji = "❌"  // Missing critical file
                    } else {
                        statusEmoji = "⚠️"  // Missing non-critical file
                    }
                    let sizeInfo = exists ? " (\(formatFileSize(fileSize)))" : ""
                    let corruptNote = isEmptyOrCorrupt ? " [EMPTY/CORRUPT]" : ""
                    print("  \(statusEmoji) \(file)\(sizeInfo)\(corruptNote)")
                }
                
                // Size verification
                let sizePercentage = expectedSize > 0 ? Double(actualTotalSize) / Double(expectedSize) * 100.0 : 0
                // Check if size is valid (>=95% of expected)
                let hasValidSize = sizePercentage >= 95.0
                
                if verbose {
                    print("\n📊 Size verification:")
                    print("  - Expected size: \(formatFileSize(Int64(expectedSize)))")
                    print("  - Actual size: \(formatFileSize(actualTotalSize))")
                    print("  - Completeness: \(String(format: "%.1f", sizePercentage))%")
                    
                    // Check if size is significantly different from expected
                    let sizeStatusEmoji = hasValidSize ? "✅" : "⚠️"
                    print("  - Size validation: \(sizeStatusEmoji) \(hasValidSize ? "Valid" : "Incomplete")")
                }
                
                // Update model size if it's significantly different from expected
                // (either too large or too small)
                if abs(sizePercentage - 100.0) > 5.0, let model = self.getModel(for: modelId) {
                    // If actual size is more than 5% different from expected
                    print("📏 Updating model size from \(formatFileSize(Int64(expectedSize))) to \(formatFileSize(actualTotalSize))")
                    DispatchQueue.main.async {
                        model.size = Int(actualTotalSize)
                        
                        // Update UserDefaults to persist the correct size
                        self.updateCustomModelsInUserDefaults()
                    }
                }
                
                // Update hasPlaceholders if there are empty/missing weight files
                if !emptyWeightFiles.isEmpty || !missingWeightFiles.isEmpty {
                    if let model = self.getModel(for: modelId) {
                        print("⚠️ Model \(modelId) has \(emptyWeightFiles.count) empty and \(missingWeightFiles.count - emptyWeightFiles.count) missing weight files - setting hasPlaceholders to true")
                        DispatchQueue.main.async {
                            model.hasPlaceholders = true
                        }
                    }
                }

                // CRITICAL CHECK for monolithic models: the .mlmodelc directory MUST exist with valid weights
                // If we didn't find valid weights via either zip extraction or direct directory, fail verification
                if modelConfig.isMonolithic && !foundValidMonolithicWeights {
                    print("\n❌ MONOLITHIC MODEL VERIFICATION FAILED:")
                    print("   The monolithic model directory '\(monolithicModelName ?? "unknown")' is missing or has invalid weights.")
                    print("   💡 Please download the model to get the monolithic model files.")
                    if let model = self.getModel(for: modelId) {
                        DispatchQueue.main.async {
                            model.hasPlaceholders = true
                        }
                    }
                    return false
                }

                // Summary of verification results
                if missingFiles.isEmpty && emptyWeightFiles.isEmpty && hasValidSize {
                    print("\n✅ Model verification SUCCESSFUL: All files present and valid size.")
                    return true
                } else if missingWeightFiles.isEmpty && emptyWeightFiles.isEmpty && hasValidSize {
                    print("\n⚠️ Model verification PARTIALLY SUCCESSFUL: Missing some non-critical files, but all weight files present and size looks good.")
                    print("   Missing files:")
                    for file in missingFiles {
                        print("   - \(file)")
                    }
                    return true
                } else if missingWeightFiles.isEmpty && emptyWeightFiles.isEmpty && !hasValidSize {
                    print("\n⚠️ Model verification PARTIALLY SUCCESSFUL: All required files present but total size (\(formatFileSize(actualTotalSize))) is less than expected (\(formatFileSize(Int64(expectedSize)))).")
                    return true
                } else {
                    print("\n❌ Model verification FAILED:")
                    if !emptyWeightFiles.isEmpty {
                        print("   Empty/corrupt weight files (0 KB):")
                        for file in emptyWeightFiles {
                            print("   - \(file)")
                        }
                    }
                    let actualMissingFiles = missingWeightFiles.filter { !emptyWeightFiles.contains($0) }
                    if !actualMissingFiles.isEmpty {
                        print("   Missing weight files:")
                        for file in actualMissingFiles {
                            print("   - \(file)")
                        }
                    }
                    if !hasValidSize {
                        print("   - Size mismatch: Expected \(formatFileSize(Int64(expectedSize))), got \(formatFileSize(actualTotalSize)) (\(String(format: "%.1f", sizePercentage))%)")
                    }
                    return false
                }
            } catch {
                print("Error parsing meta.yaml: \(error)")
            }
        }
        
        // If we couldn't verify with meta.yaml, check for essential model files
        // Look for .mlmodelc directories which are essential for CoreML models
        do {
            let contents = try fileManager.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            let mlmodelcDirs = contents.filter { $0.pathExtension == "mlmodelc" }
            
            if !mlmodelcDirs.isEmpty && (configJsonExists || tokenizerJsonExists) {
                print("Found \(mlmodelcDirs.count) .mlmodelc directories and essential files")
                
                // Check for weight.bin files in each .mlmodelc directory
                // Minimum size for weight.bin files (1KB) - anything smaller is likely corrupt/placeholder
                let minimumWeightFileSize: Int64 = 1024
                var hasAllWeightFiles = true
                var weightFilesMissing: [String] = []
                var emptyWeightFilesInFallback: [String] = []

                for mlmodelcDir in mlmodelcDirs {
                    let weightsDir = mlmodelcDir.appendingPathComponent("weights")
                    let weightBinPath = weightsDir.appendingPathComponent("weight.bin")

                    let weightBinExists = fileManager.fileExists(atPath: weightBinPath.path)
                    let dirName = mlmodelcDir.lastPathComponent

                    // Get size if file exists
                    var fileSize: Int64 = 0
                    if weightBinExists {
                        fileSize = getFileSize(at: weightBinPath.path)
                        actualTotalSize += fileSize
                        fileSizes["\(dirName)/weights/weight.bin"] = fileSize
                    }

                    // Check if file is empty/corrupt
                    let isEmptyOrCorrupt = weightBinExists && fileSize < minimumWeightFileSize

                    // Print with size information
                    if verbose {
                        let sizeInfo = weightBinExists ? " (\(formatFileSize(fileSize)))" : ""
                        let corruptNote = isEmptyOrCorrupt ? " [EMPTY/CORRUPT]" : ""
                        let statusEmoji = isEmptyOrCorrupt ? "❌" : (weightBinExists ? "✅" : "❌")
                        print("  \(statusEmoji) \(dirName)/weights/weight.bin\(sizeInfo)\(corruptNote)")
                    }

                    if !weightBinExists {
                        hasAllWeightFiles = false
                        weightFilesMissing.append("\(dirName)/weights/weight.bin")
                        allRequiredFilesExist = false
                    } else if isEmptyOrCorrupt {
                        hasAllWeightFiles = false
                        emptyWeightFilesInFallback.append("\(dirName)/weights/weight.bin")
                        weightFilesMissing.append("\(dirName)/weights/weight.bin")
                        allRequiredFilesExist = false
                    }
                }

                // Update hasPlaceholders if there are empty/missing weight files
                if !emptyWeightFilesInFallback.isEmpty || !weightFilesMissing.isEmpty {
                    if let model = self.getModel(for: modelId) {
                        DispatchQueue.main.async {
                            model.hasPlaceholders = true
                        }
                    }
                }
                
                // Size verification
                let sizePercentage = expectedSize > 0 ? Double(actualTotalSize) / Double(expectedSize) * 100.0 : 0
                // Check if size is valid (>=95% of expected)
                let hasValidSize = sizePercentage >= 95.0
                
                if verbose {
                    print("\n📊 Size verification:")
                    print("  - Expected size: \(formatFileSize(Int64(expectedSize)))")
                    print("  - Actual size: \(formatFileSize(actualTotalSize))")
                    print("  - Completeness: \(String(format: "%.1f", sizePercentage))%")
                    
                    // Check if size is significantly different from expected
                    let sizeStatusEmoji = hasValidSize ? "✅" : "⚠️"
                    print("  - Size validation: \(sizeStatusEmoji) \(hasValidSize ? "Valid" : "Incomplete")")
                }
                
                if hasAllWeightFiles && hasValidSize {
                    if verbose {
                        print("\n✅ Model verification SUCCESSFUL: All weight files present and valid size.")
                    }
                    return true
                } else if hasAllWeightFiles && !hasValidSize {
                    if verbose {
                        print("\n⚠️ Model verification PARTIALLY SUCCESSFUL: All weight files present but total size (\(formatFileSize(actualTotalSize))) is less than expected (\(formatFileSize(Int64(expectedSize)))).")
                    }
                    return true
                } else {
                    if verbose {
                        print("\n❌ Model verification FAILED:")
                        if !emptyWeightFilesInFallback.isEmpty {
                            print("   Empty/corrupt weight files (0 KB):")
                            for file in emptyWeightFilesInFallback {
                                print("   - \(file)")
                            }
                        }
                        let actualMissingFiles = weightFilesMissing.filter { !emptyWeightFilesInFallback.contains($0) }
                        if !actualMissingFiles.isEmpty {
                            print("   Missing weight files:")
                            for file in actualMissingFiles {
                                print("   - \(file)")
                            }
                        }
                        if !hasValidSize {
                            print("   - Size mismatch: Expected \(formatFileSize(Int64(expectedSize))), got \(formatFileSize(actualTotalSize)) (\(String(format: "%.1f", sizePercentage))%)")
                        }
                    }
                    return false
                }
            } else {
                if verbose {
                    print("❌ Model verification FAILED: No .mlmodelc directories found or missing essential files")
                }
                return false
            }
        } catch {
            if verbose {
                print("Error checking for .mlmodelc directories: \(error)")
            }
        }
        
        if verbose {
            print("❌ Model verification FAILED: Could not verify model files")
        }
        return allRequiredFilesExist
    }

    /// Loads a model for inference when user clicks the Load button
    @MainActor
    public func loadModel(_ model: Model) {
        print("🔄 User requested to load model: \(model.name)")
        
        // First check if model is downloaded
        if !model.isDownloaded {
            print("⚠️ Cannot load model that is not downloaded: \(model.id)")
            return
        }
        
        // Select the model first (this will update the UI)
        selectModel(model)
        
        // Get the model path
        let modelPath = getModelPath(for: model.id)
        print("📂 Model path for loading: \(modelPath.path)")
        
        // Check if model is already loading - if so, cancel it first
        if InferenceService.shared.isLoadingModel {
            // If current model is already loading, cancel it before starting new load
            InferenceService.shared.cancelModelLoading(reason: .startingNewModel)
            
            // Add a small delay to ensure cancellation is complete
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
                // Now start the actual loading
                self?.startModelLoading(model: model, modelPath: modelPath)
            }
        } else {
            // Start loading immediately if nothing is loading
            startModelLoading(model: model, modelPath: modelPath)
        }
    }
    
    // Helper method to start the actual model loading
    @MainActor
    private func startModelLoading(model: Model, modelPath: URL) {
        // Ensure any existing model is unloaded before starting to load a new one
        InferenceService.shared.unloadModel()
        
        // Start the loading process with high priority
        Task(priority: .userInitiated) {
            do {
                print("🚀 Starting model loading for: \(model.id) (0%)")
                
                // Set loading state in InferenceService
                InferenceService.shared.loadingStatus = "Starting model loading..."
                InferenceService.shared.isLoadingModel = true
                InferenceService.shared.loadingProgress = 0.05
                
                // Post notification that model loading has started
                NotificationCenter.default.post(
                    name: Notification.Name("ModelLoadingStarted"),
                    object: model.id
                )
                
                // Start loading the model
                try await InferenceService.shared.loadModel(modelId: model.id, from: modelPath)
                
                print("✅ Model loaded successfully: \(model.id) (100%)")
            } catch {
                print("❌ MS.1 Error loading model: \(error.localizedDescription)")
                
                // Don't post failure notifications for cancellation errors
                if error is CancellationError {
                    print("🔄 Model loading was cancelled - not posting error notification")
                    
                    // Just reset loading state without posting error
                    InferenceService.shared.isLoadingModel = false
                    InferenceService.shared.loadingProgress = 0
                    InferenceService.shared.loadingStatus = ""
                    return
                }
                
                // Reset loading state
                InferenceService.shared.isLoadingModel = false
                InferenceService.shared.loadingProgress = 0
                InferenceService.shared.loadingStatus = "Error: \(error.localizedDescription)"
                
                // Post notification that model loading failed (only for non-cancellation errors)
                NotificationCenter.default.post(
                    name: Notification.Name("ModelLoadingFailed"),
                    object: model.id,
                    userInfo: ["error": error.localizedDescription]
                )
            }
        }
    }
    
    /// Cancels an ongoing model download, saving resume data for all in-progress file downloads
    public func cancelDownload(modelId: String) {
        print("🛑 Cancelling download for model: \(modelId)")

        // Stop any activity indicators
        stopDownloadActivityIndicator(for: modelId)

        var cancelledTasks = 0

        // Thread-safe access to get tasks to cancel
        let fileTasksToCancel: [(String, URLSessionDownloadTask)] = withDownloadLock {
            return Array(fileDownloadTasks.filter { $0.key.hasPrefix("\(modelId):") })
        }

        for (fileKey, task) in fileTasksToCancel {
            // Cancel with resume data by calling cancelByProducingResumeData
            task.cancel(byProducingResumeData: { [weak self] resumeData in
                guard let self = self, let resumeData = resumeData else { return }
                print("💾 Saved resume data for: \(fileKey) (\(resumeData.count) bytes)")
                self.withDownloadLock {
                    self.resumeDataStore[fileKey] = resumeData
                }
            })
            cancelledTasks += 1
        }

        // Thread-safe cleanup of task tracking
        withDownloadLock {
            for (fileKey, _) in fileTasksToCancel {
                fileDownloadTasks.removeValue(forKey: fileKey)
            }
            // Clean up task info
            let taskIdsToRemove = taskToFileInfo.filter { $0.value.modelId == modelId }.map { $0.key }
            for taskId in taskIdsToRemove {
                taskToFileInfo.removeValue(forKey: taskId)
            }
        }

        if cancelledTasks > 0 {
            print("📥 Cancelled \(cancelledTasks) file download tasks for model: \(modelId)")
        }

        // Cancel top-level download tasks (legacy system)
        if let task = downloadTasks[modelId] {
            if let downloadTask = task as? URLSessionDownloadTask {
                downloadTask.cancel(byProducingResumeData: { [weak self] resumeData in
                    guard let self = self, let resumeData = resumeData else { return }
                    print("💾 Saved resume data for main task: \(modelId) (\(resumeData.count) bytes)")
                    self.withDownloadLock {
                        self.resumeDataStore["\(modelId):main"] = resumeData
                    }
                })
                print("Download task cancelled for model: \(modelId)")
            } else if let combinedTask = task as? CombinedDownloadTask {
                combinedTask.cancel()
                print("Combined download task cancelled for model: \(modelId)")
            }

            downloadTasks.removeValue(forKey: modelId)
        }

        // Stop the watchdog timer
        stopDownloadWatchdog(for: modelId)

        // Update model status - mark as partially downloaded if we have resume data
        let hasResumeData: Bool = withDownloadLock {
            return resumeDataStore.keys.contains { $0.hasPrefix("\(modelId):") }
        }
        modelDownloadStatus[modelId] = hasResumeData ? .partiallyDownloaded : .notDownloaded
        currentDownloadingFiles[modelId] = hasResumeData ? "Download paused - can be resumed" : "Download cancelled"
        print("[DOWNLOAD] \(hasResumeData ? "⏸️ PAUSED" : "🛑 STOPPED"): \(modelId)")

        // Clear byte tracking for this model
        for key in bytesDownloadedPerFile.keys where key.hasPrefix("\(modelId):") {
            bytesDownloadedPerFile.removeValue(forKey: key)
        }
        totalBytesDownloadedForModel.removeValue(forKey: modelId)
        totalBytesExpectedForModel.removeValue(forKey: modelId)

        // Thread-safe cleanup of file completion handlers
        let handlersToCall: [(Bool) -> Void] = withDownloadLock {
            var handlers: [(Bool) -> Void] = []
            for key in fileCompletionHandlers.keys where key.hasPrefix("\(modelId):") {
                if let handler = fileCompletionHandlers[key] {
                    handlers.append(handler)
                }
                fileCompletionHandlers.removeValue(forKey: key)
            }
            return handlers
        }
        // Call handlers outside the sync block to avoid deadlock
        for handler in handlersToCall {
            handler(false)
        }

        // Clean up resources
        cleanupDownload(for: modelId)

        // Add cleanup of tracking data with a delay
        cleanupDownloadTracking(for: modelId, delay: 2.0)

        // Notify any completion handlers with failure
        completionHandlers[modelId]?(false)
        completionHandlers.removeValue(forKey: modelId)

        // Allow device to sleep
        allowSleepAfterDownload()

        print("✅ Cancel complete for model: \(modelId) (hasResumeData: \(hasResumeData))")
    }

    /// Checks if a model has resume data available for continuing an interrupted download
    public func hasResumeData(for modelId: String) -> Bool {
        return resumeDataStore.keys.contains { $0.hasPrefix("\(modelId):") }
    }

    /// Clears all resume data for a model (use before force redownload)
    public func clearResumeData(for modelId: String) {
        for key in resumeDataStore.keys where key.hasPrefix("\(modelId):") {
            resumeDataStore.removeValue(forKey: key)
        }
        print("🗑️ Cleared resume data for model: \(modelId)")
    }
    
    /// Starts an activity indicator for a model download
    private func startDownloadActivityIndicator(for modelId: String) {
        // Make sure we're not already tracking activity for this model
        stopDownloadActivityIndicator(for: modelId)
        
        // Mark as active
        isDownloadActive[modelId] = true
        
        // Create a timer that updates the activity indicator
        let timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self = self, self.isDownloadActive[modelId] == true else { return }
            
            // Get current progress
            let currentProgress = self.downloadProgress[modelId] ?? 0.01
            
            // Only pulse if progress is not changing rapidly
            if currentProgress > 0 && currentProgress < 0.99 {
                // Create a subtle pulse effect by slightly increasing and decreasing the progress
                let pulseAmount = 0.005 // Small amount to pulse
                let pulseProgress = currentProgress + (sin(Date().timeIntervalSince1970 * 5) * pulseAmount)
                
                // Update UI with pulsing effect (progress bar only, don't modify status text)
                DispatchQueue.main.async {
                    // Update the progress observers with the pulsing effect
                    self.progressObservers[modelId]?(pulseProgress)

                    // Note: Don't modify currentDownloadingFiles here - it causes UI flashing
                    // The DownloadProgressView already has a built-in ProgressView spinner
                }
            }
        }
        
        // Store the timer
        downloadActivityTimers[modelId] = timer
    }
    
    /// Stops the activity indicator for a model download
    private func stopDownloadActivityIndicator(for modelId: String) {
        isDownloadActive[modelId] = false
        downloadActivityTimers[modelId]?.invalidate()
        downloadActivityTimers[modelId] = nil
    }

    // MARK: - Download Logging Helper

    /// Returns a timestamp string for download logs (HH:mm:ss format)
    private func downloadTimestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: Date())
    }

    /// Logs a download message with timestamp
    private func logDownload(_ message: String) {
        print("[DOWNLOAD] [\(downloadTimestamp())] \(message)")
    }

    // MARK: - Download Watchdog (Stall Detection)

    /// Starts a watchdog timer to detect stalled downloads
    private func startDownloadWatchdog(for modelId: String) {
        // Stop any existing watchdog
        stopDownloadWatchdog(for: modelId)

        // Check every 30 seconds for stalled downloads
        let timer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }

            // Check if download is still active
            guard self.modelDownloadStatus[modelId] == .downloading else {
                print("[DOWNLOAD] Watchdog: Download no longer active for \(modelId)")
                self.stopDownloadWatchdog(for: modelId)
                return
            }

            // Check time since last progress update
            guard let lastUpdate = self.lastProgressUpdateTime[modelId] else {
                print("[DOWNLOAD] Watchdog: No progress recorded for \(modelId)")
                return
            }

            let timeSinceUpdate = Date().timeIntervalSince(lastUpdate)
            let currentProgress = self.maxDownloadProgress[modelId] ?? 0

            if timeSinceUpdate > 60 {
                // No progress in 60 seconds - likely stalled
                print("[DOWNLOAD] ⚠️ STALL DETECTED: No progress for \(Int(timeSinceUpdate))s on \(modelId)")
                print("[DOWNLOAD] Current progress: \(Int(currentProgress * 100))%")
                print("[DOWNLOAD] Last update: \(lastUpdate)")

                // Update status to show stall warning
                DispatchQueue.main.async {
                    let currentStatus = self.currentDownloadingFiles[modelId] ?? ""
                    if !currentStatus.contains("stalled") && !currentStatus.contains("Error") {
                        self.currentDownloadingFiles[modelId] = "Download may be stalled... (\(Int(currentProgress * 100))%)"
                    }
                }
            } else if timeSinceUpdate > 30 {
                // Warn about slow progress
                print("[DOWNLOAD] Slow: No progress for \(Int(timeSinceUpdate))s on \(modelId) (\(Int(currentProgress * 100))%)")
            }
        }

        downloadWatchdogTimers[modelId] = timer
    }

    /// Stops the download watchdog timer
    private func stopDownloadWatchdog(for modelId: String) {
        downloadWatchdogTimers[modelId]?.invalidate()
        downloadWatchdogTimers[modelId] = nil
    }
    
    /// Returns a rotating activity indicator string
    private func getActivityIndicator() -> String {
        let indicators = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        let index = Int(Date().timeIntervalSince1970 * 5) % indicators.count
        return indicators[index]
    }
    
    /// Refreshes the list of available models
    @MainActor
    public func refreshAvailableModels() {
        print("Refreshing available models list")
        // Make a temporary copy of available models to avoid modifying during iteration
        let models = self.availableModels
        
        // Update download status for each model
        for model in models {
            model.refreshDownloadStatus()
        }
        
        // Update downloaded models list
        self.downloadedModels = models.filter { $0.isDownloaded }
        
        // Notify observers that models have been refreshed
        self.objectWillChange.send()
    }
    
    // Download only specific files for a model
    public func downloadSpecificFiles(modelId: String, fileList: [String], fileProgress: @escaping (String, Double) -> Void, completion: @escaping (Bool) -> Void) {
        print("Starting selective download for model: \(modelId), files to download: \(fileList.count)")
        
        // Prevent device from sleeping during download
        preventSleepDuringDownload()
        
        // Check if file list is empty
        if fileList.isEmpty {
            print("No files to download")
            DispatchQueue.main.async {
                completion(true) // Nothing to download is technically successful
            }
            // Allow sleep if there are no files to download
            allowSleepAfterDownload()
            return
        }
        
        // Find the model
        guard let model = getModel(for: modelId) else {
            print("Error: Model with ID \(modelId) not found")
            DispatchQueue.main.async {
                completion(false)
            }
            // Allow sleep if model not found
            allowSleepAfterDownload()
            return
        }
        
        // IMPORTANT: Use the model's actual ID, not the modelId parameter
        let actualModelId = model.id
        print("Using actual model ID for selective download: \(actualModelId)")
        
        // Get the proper model directory path
        let modelDir = getModelPath(for: actualModelId)
        print("Model directory for selective download: \(modelDir.path)")
        
        // Check if we have write access before starting download
        let testFilePath = modelStorageDirectory.appendingPathComponent("write_test.txt")
        do {
            try "Test download access".write(to: testFilePath, atomically: true, encoding: .utf8)
            try fileManager.removeItem(at: testFilePath)
            print("Write access confirmed for selective download")
        } catch {
            print("ERROR: No write access to models directory for selective download: \(error)")
            DispatchQueue.main.async {
                completion(false)
            }
            return
        }
        
        // Create model directory if it doesn't exist (though it should exist for partial downloads)
        if !fileManager.fileExists(atPath: modelDir.path) {
            do {
                try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true, attributes: nil)
                print("Created model directory at: \(modelDir.path)")
            } catch {
                print("ERROR: Failed to create model directory: \(error)")
                DispatchQueue.main.async {
                    completion(false)
                }
                // Allow sleep if we fail to create directory
                allowSleepAfterDownload()
                return
            }
        }
        
        // Store progress and completion handlers
        // Note: Don't call fileProgress with generic message here - it overwrites
        // the detailed status set by URLSession delegate.
        progressObservers[actualModelId] = { _ in
            // Progress is handled by fileProgressObservers via URLSession delegate
        }
        
        fileProgressObservers[actualModelId] = fileProgress
        completionHandlers[actualModelId] = { success in
            DispatchQueue.main.async {
                // Update model status
                if let model = self.getModel(for: actualModelId) {
                    model.refreshDownloadStatus()
                    
                    // Verify model files 
                    let isValid = self.verifyModelFiles(modelId: actualModelId)
                    print("Model verification result after selective download: \(isValid)")
                    
                    if success && isValid {
                        // Update downloaded models list if successful
                        if !self.downloadedModels.contains(where: { $0.id == model.id }) {
                            self.downloadedModels.append(model)
                        }
                    }
                }
                
                // Call completion handler
                completion(success)
                
                // Clean up
                self.cleanupDownload(for: actualModelId)
            }
        }
        
        // Update model download status to downloading
        modelDownloadStatus[actualModelId] = .downloading
        
        // Check if this is a Hugging Face model
        if model.downloadURL.hasPrefix("https://huggingface.co/") || 
           model.downloadURL.hasPrefix("huggingface://") {
            print("Starting selective Hugging Face download")
            downloadPartialHuggingFaceModel(modelId: actualModelId, modelURL: model.downloadURL, filesToDownload: fileList)
            return
        }
        
        // For direct URL downloads, we don't currently support partial downloads
        // So just report failure with appropriate message
        print("Selective downloads not supported for direct URL downloads")
        DispatchQueue.main.async {
            completion(false)
        }
        // Allow sleep if we don't support this type of download
        allowSleepAfterDownload()
        return
    }
    
    private func downloadPartialHuggingFaceModel(modelId: String, modelURL: String, filesToDownload: [String]) {
        print("Downloading specific files from Hugging Face for model \(modelId)")
        print("Files to download: \(filesToDownload.joined(separator: ", "))")
        
        // Extract repo info from URL
        guard let repoInfo = extractHuggingFaceRepoInfo(from: modelURL) else {
            print("Invalid Hugging Face model URL: \(modelURL)")
            reportDownloadFailure(for: modelId)
            return
        }
        
        print("⚠️ DEBUG: Hugging Face Repository Details:")
        print("  - Owner: \(repoInfo.owner)")
        print("  - Repo: \(repoInfo.repo)")
        print("  - Base URL: https://huggingface.co/\(repoInfo.owner)/\(repoInfo.repo)/resolve/main/")
        
        // Get the model directory path
        let modelDir = getModelPath(for: modelId)
        
        // First, download meta.yaml to get proper capitalization
        print("Downloading meta.yaml first to get correct capitalization...")
        let metaYamlURL = URL(string: "https://huggingface.co/\(repoInfo.owner)/\(repoInfo.repo)/resolve/main/meta.yaml")!
        let metaYamlPath = modelDir.appendingPathComponent("meta.yaml")
        
        // Create a semaphore to wait for meta.yaml download
        let semaphore = DispatchSemaphore(value: 0)
        var metaYamlSuccess = false
        var modelConfig: ModelConfiguration?
        
        // Download meta.yaml first
        let metaYamlTask = URLSession.shared.downloadTask(with: metaYamlURL) { tempURL, response, error in
            defer { semaphore.signal() }
            
            if let error = error {
                print("Error downloading meta.yaml: \(error)")
                return
            }
            
            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode != 200 {
                print("HTTP error downloading meta.yaml: \(httpResponse.statusCode)")
                return
            }
            
            guard let tempURL = tempURL else {
                print("No temp URL for meta.yaml download")
                return
            }
            
            do {
                // Create model directory if it doesn't exist
                if !self.fileManager.fileExists(atPath: modelDir.path) {
                    try self.fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)
                }
                
                // Move meta.yaml to the model directory
                if self.fileManager.fileExists(atPath: metaYamlPath.path) {
                    try self.fileManager.removeItem(at: metaYamlPath)
                }
                try self.fileManager.moveItem(at: tempURL, to: metaYamlPath)
                print("✅ Successfully downloaded meta.yaml")
                
                // Parse the meta.yaml file to get model configuration
                let metaYamlContent = try String(contentsOf: metaYamlPath, encoding: .utf8)
                print("meta.yaml content:")
                print(metaYamlContent)
                
                // Parse the model configuration to get the correct capitalization
                modelConfig = try ModelConfiguration(from: metaYamlContent, modelPath: modelDir.path)
                if let config = modelConfig {
                    print("Model prefix with correct capitalization: \(config.modelPrefix)")
                }
                
                metaYamlSuccess = true
            } catch {
                print("Error processing meta.yaml: \(error)")
            }
        }
        
        // Start meta.yaml download
        metaYamlTask.resume()
        
        // Wait for meta.yaml download with timeout
        let timeout = DispatchTime.now() + .seconds(30)
        if semaphore.wait(timeout: timeout) == .timedOut {
            print("⚠️ Timed out waiting for meta.yaml download")
            // Continue anyway, but we might have capitalization issues
        }
        
        // First, fetch the file list from the repository to help debug missing files
        fetchRepositoryFileList(owner: repoInfo.owner, repo: repoInfo.repo) { fileList in
            print("📋 Found \(fileList.count) files in repository:")
            for file in fileList.prefix(20) {
                print("  - \(file)")
            }
            if fileList.count > 20 {
                print("  - ... and \(fileList.count - 20) more files")
            }
            
            // Check if config.json exists in the repository
            if let configPath = fileList.first(where: { $0 == "config.json" || $0.hasSuffix("/config.json") }) {
                print("✅ Found config.json in repository at path: \(configPath)")
            } else {
                // Look for alternative config files
                let configPaths = fileList.filter { $0.contains("config") && $0.hasSuffix(".json") }
                if !configPaths.isEmpty {
                    print("🔍 Found alternative config files:")
                    for path in configPaths {
                        print("  - \(path)")
                    }
                } else {
                    print("❌ No config.json or alternative config files found in repository")
                }
            }
            
            // If we have the model configuration from meta.yaml, use it to get correct capitalization
            var filesToDownloadWithCorrectCase = filesToDownload
            if let config = modelConfig {
                print("Using correct capitalization from meta.yaml...")
                
                // Get the required files with correct capitalization
                let requiredFilesWithCorrectCase = self.getRequiredFiles(from: config)
                
                // Match the files to download with the correct capitalization
                filesToDownloadWithCorrectCase = filesToDownload.map { file -> String in
                    // Try to find a matching file with correct capitalization
                    if let matchingFile = requiredFilesWithCorrectCase.first(where: { $0.lowercased() == file.lowercased() }) {
                        print("Found correct capitalization for \(file): \(matchingFile)")
                        return matchingFile
                    }
                    return file
                }
            } else if !metaYamlSuccess {
                print("Warning: meta.yaml download failed, using original capitalization which may cause issues")
            }
            
            // Continue with download process using corrected file names
            self.continuePartialDownload(modelId: modelId, modelURL: modelURL, filesToDownload: filesToDownloadWithCorrectCase, repoInfo: repoInfo)
        }
    }
    
    // Helper method to fetch file list from a Hugging Face repository
    private func fetchRepositoryFileList(owner: String, repo: String, completion: @escaping ([String]) -> Void) {
        let urlString = "https://huggingface.co/api/models/\(owner)/\(repo)/tree/main"
        print("🔍 Fetching file list from: \(urlString)")
        
        guard let url = URL(string: urlString) else {
            print("❌ Invalid URL for fetching repository file list")
            completion([])
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            if let httpResponse = response as? HTTPURLResponse {
                print("📡 Repository API response: HTTP \(httpResponse.statusCode)")
                
                if httpResponse.statusCode != 200 {
                    print("❌ Failed to fetch repository file list: HTTP \(httpResponse.statusCode)")
                    completion([])
                    return
                }
            }
            
            if let error = error {
                print("❌ Error fetching repository file list: \(error.localizedDescription)")
                completion([])
                return
            }
            
            guard let data = data else {
                print("❌ No data received from repository API")
                completion([])
                return
            }
            
            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                    var fileList: [String] = []
                    
                    for item in json {
                        if let path = item["path"] as? String, !path.hasSuffix("/") {
                            fileList.append(path)
                        }
                    }
                    
                    print("✅ Successfully parsed repository file list: \(fileList.count) files")
                    completion(fileList)
                } else {
                    print("❌ Invalid JSON format from repository API")
                    
                    // Attempt to log the response for debugging
                    if let responseString = String(data: data, encoding: .utf8) {
                        print("Response body: \(responseString.prefix(1000))...")
                    }
                    
                    completion([])
                }
            } catch {
                print("❌ Error parsing repository file list: \(error.localizedDescription)")
                completion([])
            }
        }.resume()
    }

    /// Fetches file sizes from HuggingFace tree API for a list of files
    /// Returns a dictionary mapping file paths to their sizes in bytes
    private func fetchFileSizes(owner: String, repo: String, files: [String], completion: @escaping ([String: Int64]) -> Void) {
        // HuggingFace tree API requires fetching directory by directory
        // For simplicity, we'll fetch the entire tree recursively
        let urlString = "https://huggingface.co/api/models/\(owner)/\(repo)/tree/main?recursive=true"
        print("📏 Fetching file sizes from: \(urlString)")

        guard let url = URL(string: urlString) else {
            print("❌ Invalid URL for fetching file sizes")
            completion([:])
            return
        }

        URLSession.shared.dataTask(with: url) { data, response, error in
            if let error = error {
                print("❌ Error fetching file sizes: \(error.localizedDescription)")
                completion([:])
                return
            }

            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                print("❌ Failed to fetch file sizes: HTTP error")
                completion([:])
                return
            }

            guard let data = data else {
                print("❌ No data received for file sizes")
                completion([:])
                return
            }

            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                    var fileSizes: [String: Int64] = [:]

                    for item in json {
                        if let path = item["path"] as? String,
                           let size = item["size"] as? Int64 {
                            fileSizes[path] = size
                        } else if let path = item["path"] as? String,
                                  let sizeInt = item["size"] as? Int {
                            fileSizes[path] = Int64(sizeInt)
                        }
                    }

                    print("✅ Successfully fetched sizes for \(fileSizes.count) files")
                    completion(fileSizes)
                } else {
                    print("❌ Invalid JSON format for file sizes")
                    completion([:])
                }
            } catch {
                print("❌ Error parsing file sizes: \(error.localizedDescription)")
                completion([:])
            }
        }.resume()
    }

    /// Calculates total expected bytes for a model download based on file list
    private func calculateTotalExpectedBytes(for modelId: String, files: [String], owner: String, repo: String, completion: @escaping (Int64) -> Void) {
        fetchFileSizes(owner: owner, repo: repo, files: files) { [weak self] fileSizes in
            guard let self = self else {
                completion(0)
                return
            }

            var totalBytes: Int64 = 0
            var foundSizes = 0
            var missingSizes: [String] = []

            for file in files {
                if let size = fileSizes[file] {
                    totalBytes += size
                    foundSizes += 1
                } else {
                    missingSizes.append(file)
                }
            }

            if !missingSizes.isEmpty && missingSizes.count <= 5 {
                print("⚠️ Missing sizes for \(missingSizes.count) files: \(missingSizes)")
            } else if missingSizes.count > 5 {
                print("⚠️ Missing sizes for \(missingSizes.count) files")
            }

            print("📊 Total expected bytes for \(modelId): \(totalBytes) bytes (\(Double(totalBytes) / 1_000_000.0) MB) from \(foundSizes) files")

            // Store the total expected bytes
            self.withDownloadLock {
                self.totalBytesExpectedForModel[modelId] = totalBytes
            }

            DispatchQueue.main.async {
                completion(totalBytes)
            }
        }
    }

    private func continuePartialDownload(modelId: String, modelURL: String, filesToDownload: [String], repoInfo: HuggingFaceRepoInfo) {
        let modelDir = getModelPath(for: modelId)
        var downloadedCount = 0
        let totalFiles = filesToDownload.count
        
        // Create a dispatch group to track all downloads
        // Store in instance property to prevent premature deallocation (semaphore crash)
        let downloadGroup = DispatchGroup()
        withDownloadLock { activeDownloadGroups["\(modelId):regular"] = downloadGroup }

        // Track overall success
        var overallSuccess = true
        
        // Cleanup any existing progress observers
        downloadProgressObservers[modelId] = []
        
        // Show initial progress indicator
        updateDownloadProgress(
            for: modelId,
            file: "preparation",
            progress: 0.01,
            message: "Preparing to download \(totalFiles) files..."
        )
        
        // Create directories for all files first
        for file in filesToDownload {
            let fileURL = modelDir.appendingPathComponent(file)
            let fileDirectory = fileURL.deletingLastPathComponent()
            
            do {
                if !fileManager.fileExists(atPath: fileDirectory.path) {
                    try fileManager.createDirectory(at: fileDirectory, withIntermediateDirectories: true, attributes: nil)
                }
            } catch {
                print("Error creating directory for file \(file): \(error)")
                overallSuccess = false
            }
        }
        
        // Get weight.bin files which need special handling
        let weightFiles = filesToDownload.filter { $0.contains("/weights/weight.bin") }
        print("📊 Found \(weightFiles.count) weight files that need special handling")
        
        // First, download the non-weight files
        let regularFiles = filesToDownload.filter { !$0.contains("/weights/weight.bin") }
        print("📄 Downloading \(regularFiles.count) regular files first")
        
        // Update to show starting regular files
        DispatchQueue.main.async {
            self.currentDownloadingFiles[modelId] = "Starting with \(regularFiles.count) regular files..."
            self.progressObservers[modelId]?(0.05)
        }
        
        // Now download each regular file
        for (index, file) in regularFiles.enumerated() {
            downloadGroup.enter()
            
            // IMPORTANT: Construct the URL using the original case-sensitive versions
            // Format: https://huggingface.co/Owner/Repo/resolve/main/file
            let fileURLString = "https://huggingface.co/\(repoInfo.owner)/\(repoInfo.repo)/resolve/main/\(file)"
            print("🔗 Downloading file from: \(fileURLString)")
            
            // Immediately update UI to show which file we're about to download
            let baseProgress = 0.05 // Start at 5%
            let regularFilesPortionOfProgress = 0.5 // Regular files are 50% of the total progress
            let currentProgress = baseProgress + regularFilesPortionOfProgress * (Double(index) / Double(max(1, regularFiles.count)))
            
            DispatchQueue.main.async {
                self.downloadProgress[modelId] = currentProgress
                self.currentDownloadingFiles[modelId] = "⬇️ \(file) (\(index+1)/\(regularFiles.count))..."
                self.progressObservers[modelId]?(currentProgress)
                if let fileProgress = self.fileProgressObservers[modelId] {
                    fileProgress(file, currentProgress)
                }
            }
            
            print("⬇️ Downloading regular file: \(file)")
            print("  • Full URL: \(fileURLString)")
            
            // Create destination URL
            let destinationURL = modelDir.appendingPathComponent(file)
            
            // Create a download task with a custom timeout for large files
            let config = URLSessionConfiguration.default
            config.timeoutIntervalForResource = 300 // 5 minutes timeout for regular files
            let session = URLSession(configuration: config)
            
            let task = session.dataTask(with: URL(string: fileURLString)!) { data, response, error in
                defer {
                    downloadGroup.leave()
                }
                
                // Handle HTTP response
                if let httpResponse = response as? HTTPURLResponse {
                    print("📡 Response for \(file): HTTP \(httpResponse.statusCode)")
                    
                    if httpResponse.statusCode == 404 {
                        print("❌ FILE NOT FOUND (404): \(file)")
                        
                        // For missing files that aren't critical, create placeholders
                        if !file.contains("/weights/weight.bin") && !file.contains("config.json") && !file.contains("tokenizer.json") {
                            print("Creating placeholder for non-critical file: \(file)")
                            
                            let fileURL = modelDir.appendingPathComponent(file)
                            let fileDir = fileURL.deletingLastPathComponent()
                            
                            do {
                                if !self.fileManager.fileExists(atPath: fileDir.path) {
                                    try self.fileManager.createDirectory(at: fileDir, withIntermediateDirectories: true)
                                }
                                
                                // Create an empty file as placeholder
                                try Data().write(to: fileURL)
                                print("Created placeholder file for: \(file)")
                            } catch {
                                print("Error creating placeholder: \(error)")
                            }
                        }
                        
                        // Update progress tracking
                        downloadedCount += 1
                        
                        DispatchQueue.main.async {
                            let fileName = (file as NSString).lastPathComponent
                            self.currentDownloadingFiles[modelId] = "Missing file: \(fileName)"
                        }
                        return
                    }
                }
                
                // Handle potential errors
                if let error = error {
                    print("❌ Error downloading \(file): \(error.localizedDescription)")
                    overallSuccess = false
                    return
                }
                
                // Ensure we have valid data
                guard let data = data else {
                    print("❌ No data received for \(file)")
                    overallSuccess = false
                    return
                }
                
                // Try to save the file
                do {
                    // Ensure parent directory exists
                    let parentDir = destinationURL.deletingLastPathComponent()
                    if !self.fileManager.fileExists(atPath: parentDir.path) {
                        try self.fileManager.createDirectory(at: parentDir, withIntermediateDirectories: true)
                    }
                    
                    // Write the data to the destination file
                    try data.write(to: destinationURL)
                    
                    print("✅ Successfully downloaded \(file)")
                    downloadedCount += 1
                    
                    // Update UI
                    DispatchQueue.main.async {
                        let fileName = (file as NSString).lastPathComponent
                        self.currentDownloadingFiles[modelId] = "Downloaded: \(fileName)"
                    }
                } catch {
                    print("❌ Error saving file \(file): \(error.localizedDescription)")
                    overallSuccess = false
                }
            }
            
            task.resume()
        }
        
        // Wait for regular files to complete
        downloadGroup.notify(queue: .main) { [weak self] in
            guard let self = self else { return }
            print("✅ Regular files download complete. Now downloading weight files...")

            // DELAYED cleanup - wait for internal semaphore operations to complete
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                self.withDownloadLock { _ = self.activeDownloadGroups.removeValue(forKey: "\(modelId):regular") }
            }

            // Now download the weight files
            DispatchQueue.main.async {
                self.currentDownloadingFiles[modelId] = "Starting \(weightFiles.count) weight files..."
                self.downloadProgress[modelId] = 0.55 // Regular files are done (50% + 5% initial)
                self.progressObservers[modelId]?(0.55)
            }

            // Store in instance property to prevent premature deallocation (semaphore crash)
            let weightDownloadGroup = DispatchGroup()
            self.withDownloadLock { self.activeDownloadGroups["\(modelId):weights"] = weightDownloadGroup }
            
            // Download each weight file
            for (index, weightFile) in weightFiles.enumerated() {
                // IMPORTANT: Construct the URL using the original case-sensitive versions
                // Format: https://huggingface.co/Owner/Repo/resolve/main/weightFile
                let fileURLString = "https://huggingface.co/\(repoInfo.owner)/\(repoInfo.repo)/resolve/main/\(weightFile)"
                
                // Update UI to show which weight file we're about to download
                DispatchQueue.main.async {
                    self.currentDownloadingFiles[modelId] = "⬇️ Weight file: \(weightFile) (\(index+1)/\(weightFiles.count))"
                    // Progress will be updated by the downloadLFSFile method
                }
                
                print("⬇️ Downloading weight file: \(weightFile)")
                print("  • Full URL: \(fileURLString) (case-sensitive)")
                
                // Create destination URL
                let destinationURL = modelDir.appendingPathComponent(weightFile)
                
                // Download the weight file with special LFS handling
                self.downloadLFSFile(
                    url: fileURLString, 
                    destination: destinationURL, 
                    modelId: modelId, 
                    fileName: weightFile,
                    downloadGroup: weightDownloadGroup
                )
            }
            
            // Wait for weight files to complete
            weightDownloadGroup.notify(queue: .main) { [weak self] in
                guard let self = self else { return }
                print("All downloads completed, including weight files. Success: \(overallSuccess)")

                // DELAYED cleanup - wait for internal semaphore operations to complete
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    self.withDownloadLock { _ = self.activeDownloadGroups.removeValue(forKey: "\(modelId):weights") }
                }
                
                // Mark final progress
                self.updateDownloadProgress(
                    for: modelId,
                    file: "completion",
                    progress: 1.0,
                    message: "✅ Completed downloading all files"
                )
                
                // Call completion handler
                self.completionHandlers[modelId]?(overallSuccess)
                
                // Update model status
                if let model = self.getModel(for: modelId) {
                    model.refreshDownloadStatus()
                    
                    // Verify model files 
                    let isValid = self.verifyModelFiles(modelId: modelId)
                    print("Model verification result after download: \(isValid)")
                    
                    if overallSuccess && isValid {
                        // Update downloaded models list if successful
                        if !self.downloadedModels.contains(where: { $0.id == model.id }) {
                            self.downloadedModels.append(model)
                        }
                    }
                }
                
                // Clean up progress observers
                self.downloadProgressObservers[modelId] = []
                
                // Allow sleep after download
                self.allowSleepAfterDownload()
            }
        }
    }
    
    // Helper method to download LFS files
    private func downloadLFSFile(url: String, destination: URL, modelId: String, fileName: String, downloadGroup: DispatchGroup? = nil, completionHandler: ((Bool) -> Void)? = nil) {
        // Enter the dispatch group if provided
        if let group = downloadGroup {
            group.enter()
        }
        
        guard let lfsURL = URL(string: url) else {
            print("❌ Invalid LFS URL: \(url)")
            downloadGroup?.leave()
            completionHandler?(false)
            return
        }
        
        // Update progress to show LFS download is starting
        updateDownloadProgress(
            for: modelId,
            file: fileName,
            progress: 0.01,
            message: "⬇️ Starting download: \(fileName) (large file)"
        )
        
        // Create a configuration with longer timeouts for large files
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 1800 // 30 minutes for large weight files
        config.timeoutIntervalForRequest = 60 // 60 seconds for initial connection
        let session = URLSession(configuration: config)
        
        // Create a download task with progress reporting
        let task = session.downloadTask(with: lfsURL) { tempURL, response, error in
            defer {
                // Always leave the dispatch group when done
                downloadGroup?.leave()
            }
            
            // Handle errors
            if let error = error {
                print("❌ Error downloading LFS file: \(error.localizedDescription)")
                completionHandler?(false)
                self.updateDownloadProgress(
                    for: modelId,
                    file: fileName,
                    progress: 0.0,
                    message: "❌ Failed: \(fileName) - \(error.localizedDescription)"
                )
                return
            }
            
            // Check HTTP response
            if let httpResponse = response as? HTTPURLResponse {
                print("📡 LFS file HTTP response: \(httpResponse.statusCode)")
                
                if httpResponse.statusCode != 200 {
                    print("❌ HTTP Error downloading LFS file: \(httpResponse.statusCode)")
                    completionHandler?(false)
                    self.updateDownloadProgress(
                        for: modelId,
                        file: fileName,
                        progress: 0.0,
                        message: "❌ Failed: \(fileName) - HTTP \(httpResponse.statusCode)"
                    )
                    return
                }
                
                // Log content length for debugging
                let contentLength = httpResponse.expectedContentLength
                if contentLength > 0 {
                    print("📦 LFS file size: \(ByteCountFormatter.string(fromByteCount: contentLength, countStyle: .file))")
                }
            }
            
            // Ensure we have a temp URL
            guard let tempURL = tempURL else {
                print("❌ No temporary URL for LFS file")
                completionHandler?(false)
                self.updateDownloadProgress(
                    for: modelId,
                    file: fileName,
                    progress: 0.0,
                    message: "❌ Failed: \(fileName) - No temp file"
                )
                return
            }
            
            // Try to move the file to destination
            do {
                // Ensure parent directory exists
                let parentDir = destination.deletingLastPathComponent()
                if !self.fileManager.fileExists(atPath: parentDir.path) {
                    try self.fileManager.createDirectory(at: parentDir, withIntermediateDirectories: true)
                }
                
                if self.fileManager.fileExists(atPath: destination.path) {
                    try self.fileManager.removeItem(at: destination)
                }
                
                try self.fileManager.moveItem(at: tempURL, to: destination)
                print("✅ Successfully downloaded LFS file: \(fileName)")
                
                // Update UI to show success
                self.updateDownloadProgress(
                    for: modelId,
                    file: fileName,
                    progress: 1.0,
                    message: "✅ Completed: \(fileName)"
                )
                
                completionHandler?(true)
            } catch {
                print("❌ Error saving LFS file: \(error.localizedDescription)")
                completionHandler?(false)
                
                self.updateDownloadProgress(
                    for: modelId,
                    file: fileName,
                    progress: 0.0,
                    message: "❌ Failed to save: \(fileName) - \(error.localizedDescription)"
                )
            }
        }
        
        // Set up progress reporting
        let progressObserver = task.progress.observe(\.fractionCompleted) { progress, _ in
            // For weight files, we'll update the UI more frequently
            // Debug print to confirm progress updates are happening
            print("📊 Download progress for \(fileName): \(Int(progress.fractionCompleted * 100))%")
            
            // Always use a non-zero progress value to make progress visible in UI
            let visibleProgress = max(0.01, progress.fractionCompleted)
            
            // Update the UI with progress
            self.updateDownloadProgress(
                for: modelId,
                file: fileName,
                progress: visibleProgress,
                message: "📥 Downloading: \(fileName) - \(Int(visibleProgress * 100))%"
            )
        }
        
        // Store the progress observer to prevent it from being deallocated
        self.downloadProgressObservers[modelId, default: []].append(progressObserver)
        
        task.resume()
        
        print("🚀 Started LFS download for: \(fileName)")
    }
    
    public func ensureConfigurationFile(modelId: String, modelDir: URL, sourceURL: String? = nil) {
        // Check if meta.yaml exists
        let metaYamlPath = modelDir.appendingPathComponent("meta.yaml")
        if !fileManager.fileExists(atPath: metaYamlPath.path) {
            // Create a basic meta.yaml file
            
            // Extract a reasonable prefix from the model ID
            let modelPrefix = modelId.lowercased().contains("llama") ? "llama" :
                              modelId.lowercased().contains("mistral") ? "mistral" :
                              modelId.lowercased().contains("phi") ? "phi" :
                              modelId.lowercased().contains("deepseek") ? "deepseek" :
                              modelId.lowercased().split(separator: "-").first.map(String.init) ?? "model"
            
            var content = """
            model_info:
              id: \(modelId)
              format: gguf
              version: 1.0
            """
            
            // Add source URL if provided
            if let url = sourceURL, !url.isEmpty {
                content += "\n  source_url: \(url)"
            }
            
            content += """
            
            parameters:
              model_prefix: \(modelPrefix)
              context_length: 2048
              batch_size: 64
            
            files:
              - tokenizer.json
              - config.json
              - model.gguf
            """
            
            do {
                try content.write(to: metaYamlPath, atomically: true, encoding: .utf8)
                print("Created meta.yaml for model: \(modelId)")
            } catch {
                print("Error creating meta.yaml: \(error)")
            }
        } else {
            // meta.yaml exists, check if it contains source_url and update if needed
            if let sourceURL = sourceURL, !sourceURL.isEmpty {
                do {
                    var yamlContent = try String(contentsOf: metaYamlPath, encoding: .utf8)
                    
                    // Check if source_url already exists
                    if !yamlContent.contains("source_url") {
                        // Insert source_url after the id line
                        let lines = yamlContent.components(separatedBy: .newlines)
                        var updatedLines = [String]()
                        
                        for line in lines {
                            updatedLines.append(line)
                            if line.contains("id:") {
                                updatedLines.append("  source_url: \(sourceURL)")
                            }
                        }
                        
                        yamlContent = updatedLines.joined(separator: "\n")
                        try yamlContent.write(to: metaYamlPath, atomically: true, encoding: .utf8)
                        print("Updated meta.yaml with source URL for model: \(modelId)")
                    }
                    
                    // Also check if batch_size exists, add if missing
                    if !yamlContent.contains("batch_size") {
                        // Insert batch_size in the parameters section
                        let lines = yamlContent.components(separatedBy: .newlines)
                        var updatedLines = [String]()
                        var inParametersSection = false
                        var batchSizeAdded = false
                        
                        for line in lines {
                            updatedLines.append(line)
                            
                            if line.contains("parameters:") {
                                inParametersSection = true
                            } else if inParametersSection && !batchSizeAdded && (line.contains("context_length:") || line.contains("model_prefix:")) {
                                // Add batch_size after an existing parameter
                                if !updatedLines.last!.hasSuffix("batch_size: 64") {
                                    updatedLines.append("  batch_size: 64")
                                    batchSizeAdded = true
                                }
                            } else if line.contains("files:") {
                                // We've reached the end of parameters section
                                if inParametersSection && !batchSizeAdded {
                                    // Add batch_size just before files section
                                    updatedLines.insert("  batch_size: 64", at: updatedLines.count - 1)
                                }
                                inParametersSection = false
                            }
                        }
                        
                        // If we never found a good spot to add it, add it at the end of the file
                        if inParametersSection && !batchSizeAdded {
                            updatedLines.append("  batch_size: 64")
                        }
                        
                        yamlContent = updatedLines.joined(separator: "\n")
                        try yamlContent.write(to: metaYamlPath, atomically: true, encoding: .utf8)
                        print("Updated meta.yaml with batch_size parameter for model: \(modelId)")
                    }
                } catch {
                    print("Error updating meta.yaml: \(error)")
                }
            }
        }
    }
    
    // New helper method to try alternative locations for config.json
    private func tryAlternativeConfigJsonLocations(modelId: String, repoInfo: HuggingFaceRepoInfo, destinationURL: URL) {
        let alternativeURLs = [
            "https://huggingface.co/\(repoInfo.owner)/\(repoInfo.repo)/raw/main/config.json",
            "https://huggingface.co/\(repoInfo.owner)/\(repoInfo.repo)/raw/main/config.json",
            "https://huggingface.co/\(repoInfo.owner)/\(repoInfo.repo)/tree/main/config.json",
        ]
        
        print("🔄 Trying \(alternativeURLs.count) alternative locations for config.json")
        
        var attemptIndex = 0
        tryNextURL()
        
        func tryNextURL() {
            if attemptIndex >= alternativeURLs.count {
                print("❌ All alternative config.json locations failed")
                
                // Create a placeholder config.json
                let placeholderConfig = """
                {
                    "architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                    "torch_dtype": "float16",
                    "placeholder": true,
                    "note": "This is an auto-generated placeholder config.json file"
                }
                """
                
                do {
                    try placeholderConfig.write(to: destinationURL, atomically: true, encoding: .utf8)
                    print("📝 Created placeholder config.json file")
                } catch {
                    print("❌ Error creating placeholder config.json: \(error)")
                }
                
                return
            }
            
            let urlString = alternativeURLs[attemptIndex]
            print("🔍 Trying alternative config URL #\(attemptIndex + 1): \(urlString)")
            
            let task = URLSession.shared.dataTask(with: URL(string: urlString)!) { data, response, error in
                if let httpResponse = response as? HTTPURLResponse {
                    print("📡 Response for alternative config #\(attemptIndex + 1): HTTP \(httpResponse.statusCode)")
                    
                    if httpResponse.statusCode == 200, let data = data {
                        do {
                            try data.write(to: destinationURL)
                            print("✅ Successfully downloaded config.json from alternative location #\(attemptIndex + 1)")
                            return
                        } catch {
                            print("❌ Error saving config.json from alternative location: \(error)")
                        }
                    }
                }
                
                // Try next URL
                attemptIndex += 1
                tryNextURL()
            }
            
            task.resume()
        }
    }
    
    // Helper struct to store Hugging Face repository information
    private struct HuggingFaceRepoInfo {
        // Original case-sensitive components (for URL construction)
        var owner: String
        var repo: String
        
        // Lowercase versions (for ID generation and path construction)
        var ownerLowercase: String
        var repoLowercase: String
        
        init(owner: String, repo: String) {
            self.owner = owner  // Preserve original case
            self.repo = repo    // Preserve original case
            self.ownerLowercase = owner.lowercased()
            self.repoLowercase = repo.lowercased()
        }
    }
    
    // Extract owner and repo from a Hugging Face URL or identifier
    private func extractHuggingFaceRepoInfo(from url: String) -> HuggingFaceRepoInfo? {
        var owner = ""
        var repo = ""
        
        // Handle different formats of Hugging Face identifiers
        if url.hasPrefix("https://huggingface.co/") {
            // Format: https://huggingface.co/owner/repo
            let components = url.dropFirst("https://huggingface.co/".count).components(separatedBy: "/")
            if components.count >= 2 {
                owner = components[0]
                repo = components[1]
            } else {
                return nil // Invalid format
            }
        } else if url.hasPrefix("huggingface://") {
            // Format: huggingface://owner/repo
            let components = url.dropFirst("huggingface://".count).components(separatedBy: "/")
            if components.count >= 2 {
                owner = components[0]
                repo = components[1]
            } else {
                return nil // Invalid format
            }
        } else {
            // Assume format is just "owner/repo"
            let components = url.components(separatedBy: "/")
            if components.count >= 2 {
                owner = components[0]
                repo = components[1]
            } else {
                return nil // Invalid format
            }
        }
        
        // Make sure we have both owner and repo
        if owner.isEmpty || repo.isEmpty {
            return nil
        }
        
        // Return a new HuggingFaceRepoInfo struct with the original case preserved
        return HuggingFaceRepoInfo(owner: owner, repo: repo)
    }
    
    // Report a download failure and clean up
    private func reportDownloadFailure(for modelId: String) {
        DispatchQueue.main.async {
            // Update model download status
            self.modelDownloadStatus[modelId] = .failed
            
            // Call completion handler if exists
            if let completionHandler = self.completionHandlers[modelId] {
                completionHandler(false)
            }
            
            // Clean up resources
            self.cleanupDownload(for: modelId)
        }
    }
    
    // Add this method near the other helper methods
    private func updateIdleTimer() {
        #if !os(macOS)
        DispatchQueue.main.async {
            let shouldPreventSleep = self.activeDownloads > 0
            let wasAlreadyPreventing = UIApplication.shared.isIdleTimerDisabled
            
            // Only update and log if the state is changing
            if shouldPreventSleep != wasAlreadyPreventing {
                UIApplication.shared.isIdleTimerDisabled = shouldPreventSleep
                
                if shouldPreventSleep {
                    print("📱 WAKE LOCK ENABLED: Device will stay awake during download")
                } else {
                    print("📱 WAKE LOCK DISABLED: Device can sleep normally again")
                }
            }
            
            // Also log the detailed count for debugging
            print("Active download count: \(self.activeDownloads)")
        }
        #endif
    }
    
    // Add this method near the other helper methods
    private func preventSleepDuringDownload() {
        DispatchQueue.main.async {
            self.activeDownloads += 1
        }
    }
    
    // Add this method near the other helper methods
    private func allowSleepAfterDownload() {
        DispatchQueue.main.async {
            self.activeDownloads = max(0, self.activeDownloads - 1)
        }
    }
    
    /// Gets the model prefix from meta.yaml, or infers it from the model ID if not available
    public func getModelPrefix(for modelId: String) -> String {
        let modelDir = getModelPath(for: modelId)
        let metaYamlPath = modelDir.appendingPathComponent("meta.yaml")
        
        // Default prefix - only llama_ is special-cased
        let defaultPrefix = modelId.lowercased().contains("llama") ? "llama_" : "model_"
        
        // Return default if meta.yaml doesn't exist
        if !fileManager.fileExists(atPath: metaYamlPath.path) {
            print("⚠️ meta.yaml not found for model \(modelId), using default prefix: \(defaultPrefix)")
            return defaultPrefix
        }
        
        do {
            // Read meta.yaml content
            let content = try String(contentsOf: metaYamlPath, encoding: .utf8)
            
            // Look for model_prefix in the file
            let lines = content.components(separatedBy: .newlines)
            for line in lines {
                let trimmedLine = line.trimmingCharacters(in: .whitespaces)
                if trimmedLine.hasPrefix("model_prefix:") {
                    let parts = trimmedLine.components(separatedBy: "model_prefix:")
                    if parts.count >= 2 {
                        let prefix = parts[1].trimmingCharacters(in: .whitespaces)
                        if !prefix.isEmpty {
                            print("✅ Found model_prefix in meta.yaml: \(prefix)")
                            return prefix
                        }
                    }
                }
            }
            
            // Also check under parameters section
            var inParametersSection = false
            for line in lines {
                let trimmedLine = line.trimmingCharacters(in: .whitespaces)
                
                if trimmedLine == "parameters:" {
                    inParametersSection = true
                    continue
                }
                
                if inParametersSection && trimmedLine.hasPrefix("model_prefix:") {
                    let parts = trimmedLine.components(separatedBy: "model_prefix:")
                    if parts.count >= 2 {
                        let prefix = parts[1].trimmingCharacters(in: .whitespaces)
                        if !prefix.isEmpty {
                            print("✅ Found model_prefix in parameters section: \(prefix)")
                            return prefix
                        }
                    }
                }
                
                // Exit parameters section when we hit a new top-level section
                if inParametersSection && !trimmedLine.isEmpty && !trimmedLine.hasPrefix(" ") && !trimmedLine.hasPrefix("\t") {
                    inParametersSection = false
                }
            }
            
            print("⚠️ model_prefix not found in meta.yaml for model \(modelId), using default: \(defaultPrefix)")
        } catch {
            print("⚠️ Error reading meta.yaml for model \(modelId): \(error)")
        }
        
        return defaultPrefix
    }
    
    /// Gets the file path for a model file using the model prefix from meta.yaml
    public func getModelFilePath(modelId: String, fileName: String) -> URL {
        let modelDir = getModelPath(for: modelId)
        let modelPrefix = getModelPrefix(for: modelId)
        
        // Replace "MODEL_PREFIX" placeholder with actual prefix if present
        let actualFileName = fileName.replacingOccurrences(of: "MODEL_PREFIX", with: modelPrefix)
        
        return modelDir.appendingPathComponent(actualFileName)
    }
    
    // MARK: - Download Progress Tracking

    /// Add this method to provide visual progress updates during downloads
    private func updateDownloadProgress(for modelId: String, file: String, progress: Double, message: String? = nil) {
        DispatchQueue.main.async {
            // Store the current progress and update timestamp
            let displayMessage = message ?? "Downloading: \(file) - \(Int(progress * 100))%"
            
            // Print progress for debugging
            print("📊 Progress update: \(modelId) - \(file) - \(Int(progress * 100))%")
            
            // Force a minimum progress of 0.01 instead of 0.0 to make progress visible
            let visibleProgress = max(0.01, progress)
            
            // Use our renamed method to record progress and timestamp
            self.recordDownloadProgressWithTimestamp(for: modelId, fileName: displayMessage, progress: visibleProgress)
            
            // Notify progress observers with a small delay to ensure UI updates
            if let progressObserver = self.progressObservers[modelId] {
                progressObserver(visibleProgress)
            }
            
            // Update file-specific progress if needed - ensure this is always called
            if let fileProgress = self.fileProgressObservers[modelId] {
                // Debug that we're calling the file progress callback
                print("📲 Calling file progress callback for \(modelId): \(file) at \(Int(visibleProgress * 100))%")
                fileProgress(file, visibleProgress)
            } else {
                // Debug that we're missing the file progress callback
                print("⚠️ No file progress callback found for \(modelId)")
            }
        }
    }
    
    /// Clear all progress tracking for a model
    private func clearProgressTracking(for modelId: String) {
        DispatchQueue.main.async {
            // Remove all progress observers
            self.downloadProgressObservers[modelId] = []
            
            // Reset progress indicators
            self.downloadProgress[modelId] = 0
            self.currentDownloadingFiles[modelId] = nil
            self.progressObservers[modelId]?(0)
            
            print("🧹 Cleared progress tracking for \(modelId)")
        }
    }
    
    // Add method to prevent duplicate models when adding custom models
    private func modelExists(withId id: String) -> Bool {
        return availableModels.contains { $0.id == id }
    }
    
    // Update custom model addition to prevent duplicates
    public func addCustomModel(name: String? = nil, description: String? = nil, downloadURL: String, completion: @escaping (Bool, String?) -> Void) {
        print("Adding custom model from URL: \(downloadURL)")
        
        // Extract repository information from URL
        guard let repoInfo = extractHuggingFaceRepoInfo(from: downloadURL) else {
            completion(false, "Invalid Hugging Face URL format")
            return
        }
        
        // For HuggingFace repositories, use just the repository name as the ID (lowercase)
        // This preserves the repository naming convention while avoiding duplication
        let modelId = repoInfo.repoLowercase
        print("Created model ID from repository name: \(modelId)")
        print("Original case-sensitive repository name: \(repoInfo.repo)")
        
        // Check if a model with this ID already exists
        if modelExists(withId: modelId) {
            print("Model with ID \(modelId) already exists, not adding duplicate")
            
            // Find the existing model
            if let existingModel = availableModels.first(where: { $0.id == modelId }) {
                // We can't update name/description as they are 'let' properties
                // Just refresh the download status
                existingModel.refreshDownloadStatus()
                
                // Notify success but indicate it was already existing
                completion(true, "Model already exists in your library")
                return
            }
        }
        
        // Create the new model with the extracted information
        let customModel = Model(
            id: modelId,
            name: name ?? "\(repoInfo.owner)/\(repoInfo.repo)",  // Use original case for display
            description: description ?? "Custom model from Hugging Face",
            size: 0,  // We don't know the size yet
            downloadURL: downloadURL  // Keep the original URL intact
        )
        
        // Add to available models
        availableModels.append(customModel)
        
        // Refresh the model's download status
        customModel.refreshDownloadStatus()
        
        // Save the updated list of available models
        updateCustomModelsInUserDefaults()
        
        completion(true, nil)
    }
    
    // Add a method to clean up duplicate models
    private func cleanupDuplicateModels() {
        print("Cleaning up any duplicate models and fixing IDs with duplicated prefixes...")
        
        // Create a dictionary to store unique models by normalized URL (instead of ID)
        var uniqueModels: [String: Model] = [:]
        var duplicatesFound = 0
        var fixedPrefixIDs = 0
        
        // First, check for and fix any model IDs with duplicated owner prefixes
        for model in availableModels {
            // Skip processing if this isn't a HuggingFace model
            if !model.downloadURL.contains("huggingface.co/") && !model.downloadURL.contains("huggingface://") {
                continue
            }
            
            // Extract repository information from URL
            if let repoInfo = extractHuggingFaceRepoInfo(from: model.downloadURL) {
                let ownerLower = repoInfo.ownerLowercase
                
                // Check if the model ID has a duplicated owner prefix
                // Example: "anemll-anemll-deepseekr1-8b-ctx1024_0.3.0"
                let expectedPrefix = "\(ownerLower)-\(ownerLower)-"
                if model.id.lowercased().hasPrefix(expectedPrefix) {
                    // Found a model with duplicated owner prefix
                    let repoName = repoInfo.repoLowercase
                    
                    print("⚠️ Found model with duplicated owner prefix: \(model.id)")
                    print("  - Extracting repository name: \(repoName)")
                    print("  - Original case-sensitive repository name: \(repoInfo.repo)")
                    
                    // Since we can't modify the model's ID directly (it's a let property),
                    // we'll mark it as a duplicate and create a new model with the correct ID
                    
                    // Create a new model with the correct ID
                    let fixedModel = Model(
                        id: repoName,
                        name: model.name,
                        description: model.description,
                        size: model.size,
                        downloadURL: model.downloadURL
                    )
                    fixedModel.isDownloaded = model.isDownloaded
                    fixedModel.hasPlaceholders = model.hasPlaceholders
                    
                    print("✅ Created fixed model with ID: \(fixedModel.id)")
                    fixedPrefixIDs += 1
                    
                    // Add to uniqueModels to ensure we include this fixed version
            let normalizedURL = model.downloadURL.lowercased()
                .replacingOccurrences(of: "huggingface://", with: "")
                .replacingOccurrences(of: "https://huggingface.co/", with: "")
            
                    uniqueModels[normalizedURL] = fixedModel
                    continue
                }
            }
            
            // Process normal case for deduplication
            let normalizedURL = model.downloadURL.lowercased()
                .replacingOccurrences(of: "huggingface://", with: "")
                .replacingOccurrences(of: "https://huggingface.co/", with: "")
            
            // Check for duplicates
            if uniqueModels.keys.contains(where: { $0 == normalizedURL }) {
                duplicatesFound += 1
                print("Found duplicate model for URL: \(normalizedURL)")
                
                // Keep the one with the better-formatted ID if possible
                if let existingModel = uniqueModels[normalizedURL] {
                    // Prefer models with IDs that follow the standard format: owner-repo
                    if !model.id.contains("-") && existingModel.id.contains("-") {
                        // Keep the existing model with the standardized ID
                        continue
                    } else if model.id.contains("-") && !existingModel.id.contains("-") {
                        // Replace with this model as it has a better ID format
                        uniqueModels[normalizedURL] = model
                        continue
                    }
                    
                    // If both or neither have the standard format, prefer the more descriptive ID
                    if model.id.count > existingModel.id.count {
                        uniqueModels[normalizedURL] = model
                    }
                }
            } else {
                // This is the first time we've seen this URL
                uniqueModels[normalizedURL] = model
            }
        }
        
        // If we found fewer unique models than total models or fixed any IDs, update the available models list
        if uniqueModels.count < availableModels.count || fixedPrefixIDs > 0 {
            print("Found \(duplicatesFound) duplicate models and fixed \(fixedPrefixIDs) models with duplicated prefixes")
            
            // Replace available models with deduplicated list
            availableModels = Array(uniqueModels.values)
            
            // Save the deduplicated list
            saveModelsToJSON()
            
            print("✅ Updated models list saved to storage")
        } else {
            print("No duplicate models or IDs with duplicated prefixes found")
        }
    }

    // Add this immediately after the downloadModel method
    /// Forces a complete redownload of a model by removing the model directory and starting a fresh download
    public func forceRedownloadModel(
        modelId: String,
        fileProgress: @escaping (String, Double) -> Void,
        completion: @escaping (Bool) -> Void
    ) {
        print("🔄 Starting FORCED redownload for model: \(modelId)")

        // Clear any saved resume data since we're starting fresh
        clearResumeData(for: modelId)

        // Clear any cached expected file sizes
        for key in expectedFileSizes.keys where key.hasPrefix("\(modelId):") {
            expectedFileSizes.removeValue(forKey: key)
        }

        // Get the model directory path
        let modelDir = getModelPath(for: modelId)
        print("📂 Model directory: \(modelDir.path)")

        // First attempt to remove the directory if it exists
        if fileManager.fileExists(atPath: modelDir.path) {
            do {
                print("🗑️ Removing existing model directory")
                try fileManager.removeItem(at: modelDir)
                print("✅ Successfully removed model directory")
            } catch {
                print("⚠️ Failed to remove model directory: \(error.localizedDescription)")
                // Continue anyway - we'll try to download even if we couldn't remove the directory
            }
        } else {
            print("ℹ️ Model directory doesn't exist, nothing to remove")
        }

        // Introduce a small delay to ensure filesystem operations complete
        // This is especially important on iOS where file operations can be slightly delayed
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            // Now download the model with the standard download method
            self.downloadModel(modelId: modelId, fileProgress: fileProgress, completion: completion)
        }
    }

    // MARK: - Streaming File Download with Resume Support

    /// Wrapper for downloadSingleFile that handles retry with resume data
    private func downloadSingleFileWithRetry(modelId: String, fileName: String, destination: URL, completion: ((Bool) -> Void)?) {
        // Build the URL for the file
        guard let model = getModel(for: modelId) else {
            logDownload("❌ Cannot retry: Model not found for \(modelId)")
            completion?(false)
            return
        }

        // Parse HuggingFace URL from downloadURL
        var repoPath = model.downloadURL
        if repoPath.hasPrefix("https://huggingface.co/") {
            repoPath = String(repoPath.dropFirst("https://huggingface.co/".count))
        } else if repoPath.hasPrefix("huggingface://") {
            repoPath = String(repoPath.dropFirst("huggingface://".count))
        }

        // Parse owner/repo from path
        let urlComponents = repoPath.components(separatedBy: "/")
        guard urlComponents.count >= 2 else {
            logDownload("❌ Cannot retry: Invalid HuggingFace URL: \(model.downloadURL)")
            completion?(false)
            return
        }

        let owner = urlComponents[0]
        let repo = urlComponents[1]
        let fileURLString = "https://huggingface.co/\(owner)/\(repo)/resolve/main/\(fileName)"

        logDownload("🔄 Retrying download: \(fileName)")

        downloadSingleFile(urlString: fileURLString, destination: destination, modelId: modelId) { success in
            completion?(success)
        }
    }

    /// Downloads a single file using URLSessionDownloadTask for streaming (not loading into memory)
    /// Supports resume from interrupted downloads and provides proper progress tracking
    private func downloadSingleFile(urlString: String, destination: URL, modelId: String? = nil, completion: @escaping (Bool) -> Void) {
        let fileName = (destination.path as NSString).lastPathComponent

        // CRITICAL FIX: Use full relative path as key, not just fileName
        // Multiple files can have the same name (e.g., weight.bin in different directories)
        // Using just fileName causes completion handler collisions!
        let relativePath: String
        if let modelId = modelId {
            let modelDir = getModelPath(for: modelId)
            if destination.path.hasPrefix(modelDir.path) {
                // Extract relative path from model directory
                let startIndex = destination.path.index(destination.path.startIndex, offsetBy: modelDir.path.count + 1)
                relativePath = String(destination.path[startIndex...])
            } else {
                relativePath = fileName
            }
        } else {
            relativePath = fileName
        }
        let fileKey = modelId.map { "\($0):\(relativePath)" } ?? relativePath

        print("📥 Downloading file from: \(urlString)")
        print("📂 Saving to: \(destination.path)")
        print("🔑 File key: \(fileKey)")

        // Check if file already exists and validate its size
        if fileManager.fileExists(atPath: destination.path) {
            do {
                let attributes = try fileManager.attributesOfItem(atPath: destination.path)
                if let fileSize = attributes[.size] as? UInt64, fileSize > 0 {
                    // SPECIAL CHECK: weight.bin files must be at least 1MB to be valid
                    // Placeholder/corrupt weight files are often tiny (< 1KB)
                    let isWeightFile = fileName == "weight.bin" || destination.path.contains("weights/weight.bin")
                    let minimumWeightFileSize: UInt64 = 1_000_000 // 1 MB minimum for weight files

                    if isWeightFile && fileSize < minimumWeightFileSize {
                        print("⚠️ CORRUPT WEIGHT FILE: \(fileName) is only \(fileSize) bytes (< 1MB minimum), will redownload")
                        // Delete the corrupt file first
                        try? fileManager.removeItem(at: destination)
                        // Fall through to download
                    } else if let expectedSize = expectedFileSizes[fileKey], expectedSize > 0 {
                        // Check if we have expected size info to validate completeness
                        if Int64(fileSize) >= expectedSize {
                            print("✅ SKIP (already complete): \(fileName) (\(fileSize)/\(expectedSize) bytes)")
                            completion(true)
                            return
                        } else {
                            print("⚠️ PARTIAL: \(fileName) (\(fileSize)/\(expectedSize) bytes), will resume")
                            // Don't return - we'll try to resume below
                        }
                    } else if isWeightFile {
                        // Weight file with no expected size - only skip if >= 1MB
                        print("✅ SKIP (weight file >= 1MB): \(fileName) (\(fileSize) bytes)")
                        completion(true)
                        return
                    } else {
                        // No expected size info - assume file is complete if it has content
                        print("✅ SKIP (assuming complete, no expected size): \(fileName) (\(fileSize) bytes)")
                        completion(true)
                        return
                    }
                } else {
                    print("⚠️ EMPTY FILE (zero size), will redownload: \(fileName)")
                }
            } catch {
                print("⚠️ Error checking existing file: \(error.localizedDescription)")
            }
        } else {
            print("📝 File does not exist, will download: \(fileName)")
        }

        guard let url = URL(string: urlString) else {
            print("❌ Invalid URL: \(urlString)")
            setDownloadError(for: modelId, message: "Invalid URL: \(urlString)")
            completion(false)
            return
        }

        // Create parent directory if needed
        let directory = destination.deletingLastPathComponent()
        do {
            if !fileManager.fileExists(atPath: directory.path) {
                try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
                print("📁 Created directory: \(directory.path)")
            }
        } catch {
            print("❌ Error creating directory: \(error.localizedDescription)")
            setDownloadError(for: modelId, message: "Failed to create directory")
            completion(false)
            return
        }

        // Check for resume data from a previous interrupted download
        var downloadTask: URLSessionDownloadTask
        let resumeData: Data? = withDownloadLock {
            let data = resumeDataStore[fileKey]
            if data != nil {
                resumeDataStore.removeValue(forKey: fileKey)
            }
            return data
        }

        if let resumeData = resumeData {
            print("📥 Resuming download from saved data for: \(fileName)")
            downloadTask = fileDownloadSession.downloadTask(withResumeData: resumeData)
        } else {
            downloadTask = fileDownloadSession.downloadTask(with: url)
        }

        // Thread-safe storage of task tracking info
        // CRITICAL: Store relativePath (not just fileName) to correctly identify files in delegate callbacks
        let effectiveModelId = modelId ?? "unknown"
        withDownloadLock {
            fileCompletionHandlers[fileKey] = completion
            taskToFileInfo[downloadTask.taskIdentifier] = (modelId: effectiveModelId, fileName: relativePath, destination: destination)
            fileDownloadTasks[fileKey] = downloadTask
        }

        // Start the download
        downloadTask.resume()
        print("🚀 Download task started for: \(relativePath) (taskId: \(downloadTask.taskIdentifier))")
    }

    /// Legacy wrapper for backward compatibility with existing code
    private func downloadSingleFile(urlString: String, destination: URL, completion: @escaping (Bool) -> Void) {
        // Extract modelId from destination path if possible
        let pathComponents = destination.pathComponents
        var modelId: String? = nil
        if let modelsIndex = pathComponents.firstIndex(of: "Models"), modelsIndex + 1 < pathComponents.count {
            modelId = pathComponents[modelsIndex + 1]
        }
        downloadSingleFile(urlString: urlString, destination: destination, modelId: modelId, completion: completion)
    }

    /// Sets an error message for display in the UI
    private func setDownloadError(for modelId: String?, message: String) {
        guard let modelId = modelId else { return }
        DispatchQueue.main.async {
            self.downloadErrorMessages[modelId] = message
            self.currentDownloadingFiles[modelId] = "Error: \(message)"
        }
    }

    /// Gets the current error message for a model download
    public func getDownloadError(for modelId: String) -> String? {
        return downloadErrorMessages[modelId]
    }

    /// Clears the error message for a model
    public func clearDownloadError(for modelId: String) {
        downloadErrorMessages.removeValue(forKey: modelId)
    }
    
    // Helper method to try alternative URL structures for weight files
    private func getAlternativeWeightURL(originalURL: String) -> String? {
        // If the URL was trying to access weights/weight.bin inside a .mlmodelc,
        // try to access it directly through the model repository root
        if originalURL.contains(".mlmodelc/weights/weight.bin") {
            // Extract the mlmodelc name
            if let range = originalURL.range(of: "/[^/]+\\.mlmodelc/") {
                let mlmodelcPart = originalURL[range]
                let mlmodelcName = String(mlmodelcPart.dropFirst().dropLast())
                
                // Create alternative URL that might point directly to the weight file
                let baseParts = originalURL.components(separatedBy: "/resolve/")
                if baseParts.count >= 2 {
                    let baseRepo = baseParts[0]
                    let branch = baseParts[1].components(separatedBy: "/").first ?? "main"
                    return "\(baseRepo)/resolve/\(branch)/\(mlmodelcName)/weights/weight.bin"
                }
            }
        }
        return nil
    }

    // Helper method to check for missing weight files
    private func checkForMissingWeightFiles(in modelDir: URL) -> [String] {
        var missingFiles: [String] = []
        
        do {
            // Look for .mlmodelc directories
            let contents = try fileManager.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            let mlmodelcDirs = contents.filter { $0.pathExtension == "mlmodelc" }
            
            for mlmodelcDir in mlmodelcDirs {
                let weightsDir = mlmodelcDir.appendingPathComponent("weights")
                let weightBinPath = weightsDir.appendingPathComponent("weight.bin")
                
                if !fileManager.fileExists(atPath: weightBinPath.path) {
                    // Record relative path for clearer reporting
                    let dirName = mlmodelcDir.lastPathComponent
                    missingFiles.append("\(dirName)/weights/weight.bin")
                }
            }
        } catch {
            print("Error checking for missing weight files: \(error)")
        }
        
        return missingFiles
    }
    
    /// Check if a model is currently being downloaded
    func isModelDownloading(modelId: String) -> Bool {
        return currentDownloadingFiles[modelId] != nil
    }
    
    /// Remove a model from the downloadedModels array
    func removeFromDownloadedModels(modelId: String) {
        // Create a new array without this model
        var updatedDownloadedModels = [Model]()
        for downloadedModel in downloadedModels {
            if downloadedModel.id != modelId {
                updatedDownloadedModels.append(downloadedModel)
            }
        }
        
        // Replace the original array if changes were made
        if updatedDownloadedModels.count != downloadedModels.count {
            self.downloadedModels = updatedDownloadedModels
            print("Removed model \(modelId) from downloadedModels list")
        }
    }

    /// Check if a model is currently being downloaded
    func isDownloading(modelId: String) -> Bool {
        // Check if we still have an active download task
        let hasActiveTask = downloadTasks[modelId] != nil
        
        // Check if this model has any tracked downloading files
        let hasTrackedFiles = currentDownloadingFiles[modelId] != nil
        
        // Check if there's any download progress for this model
        let hasProgress = downloadProgress[modelId] != nil
        
        // If we have no active task but still have progress or tracked files, 
        // we might have a stale download state
        if !hasActiveTask && (hasTrackedFiles || hasProgress) {
            // If we're tracking files but have no active task, check how long it's been inactive
            if let lastUpdate = lastProgressUpdate[modelId],
               Date().timeIntervalSince(lastUpdate) > 5.0 { // 5 seconds without progress = stalled
                // Clear tracking data for this stalled download
                print("Cleaning up stalled download tracking for model \(modelId)")
                
                // Use DispatchQueue.main.async to avoid potential threading issues
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.cleanupDownloadTracking(for: modelId)
                }
                return false
            }
        }
        
        // Consider a model to be downloading only if we have an active task
        // AND either track files or show progress
        return hasActiveTask && (hasTrackedFiles || hasProgress)
    }
    
    // Add a private property to track last progress update time
    private var lastProgressUpdate: [String: Date] = [:]
    
    // Update this method when recording progress
    func recordDownloadProgressWithTimestamp(for modelId: String, fileName: String, progress: Double) {
        DispatchQueue.main.async {
            // Ensure progress is always visible (never zero)
            let visibleProgress = max(0.01, progress)
            
            // Store the current progress
            self.downloadProgress[modelId] = visibleProgress
            
            // Store the current file being downloaded
            self.currentDownloadingFiles[modelId] = fileName
            
            // Record timestamp of this update
            self.lastProgressUpdate[modelId] = Date()
            
            // Debug print to confirm progress is being recorded
            print("📝 Recording progress for \(modelId): \(fileName) at \(Int(visibleProgress * 100))%")
            
            // Notify progress observers - only using the callable observers
            // Note: NSKeyValueObservation objects in downloadProgressObservers are automatically notified
            // via KVO when the properties they observe change
            
            // Call the progress observer if available
            if let progressObserver = self.progressObservers[modelId] {
                progressObserver(visibleProgress)
            }
            
            // Call the file-specific progress observer if available
            if let fileProgress = self.fileProgressObservers[modelId] {
                fileProgress(fileName, visibleProgress)
            }
        }
    }
    
    /// Clean up download tracking data for a model
    private func cleanupDownloadTracking(for modelId: String, delay: TimeInterval = 0) {
        let cleanupAction = { [weak self] in
            guard let self = self else { return }
            
            DispatchQueue.main.async {
                print("Cleaning up download tracking data for \(modelId)")
                self.currentDownloadingFiles.removeValue(forKey: modelId)
                self.downloadProgress.removeValue(forKey: modelId)
                self.lastProgressUpdate.removeValue(forKey: modelId)
                self.progressObservers.removeValue(forKey: modelId)
                self.fileProgressObservers.removeValue(forKey: modelId)
            }
        }
        
        if delay > 0 {
            // Schedule cleanup after the specified delay
            DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: cleanupAction)
        } else {
            // Execute cleanup immediately
            cleanupAction()
        }
    }

    // Helper class to manage multiple download tasks as one
    class CombinedDownloadTask: NSObject {
        let tasks: [URLSessionDownloadTask]
        
        init(tasks: [URLSessionDownloadTask]) {
            self.tasks = tasks
            super.init()
        }
        
        func cancel() {
            for task in tasks {
                task.cancel()
            }
        }
        
        func resume() {
            for task in tasks {
                task.resume()
            }
        }
    }

    /// Performs detailed size verification on a model
    public func verifyModelSizes(modelId: String) -> (isValid: Bool, totalSize: Int64, sizeInfo: String) {
        let modelDir = getModelPath(for: modelId)
        var fileCount = 0
        var totalSize: Int64 = 0
        var largestFiles: [(name: String, size: Int64)] = []
        var componentSizes: [String: Int64] = [:]
        var output = ""
        
        guard let model = getModel(for: modelId),
              fileManager.fileExists(atPath: modelDir.path) else {
            return (false, 0, "Model directory doesn't exist")
        }
        
        // Expected model size from metadata
        let expectedSize = Int64(model.size)
        
        // Helper function to format file sizes
        func formatFileSize(_ size: Int64) -> String {
            let formatter = ByteCountFormatter()
            formatter.allowedUnits = [.useGB, .useMB, .useKB]
            formatter.countStyle = .file
            return formatter.string(fromByteCount: size)
        }
        
        // Helper function to recursively scan directory
        func scanDirectory(_ url: URL, component: String? = nil) {
            do {
                let contents = try fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey])
                for item in contents {
                    let resourceValues = try item.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
                    
                    if resourceValues.isDirectory == true {
                        // Track component sizes (mlmodelc directories)
                        let dirName = item.lastPathComponent
                        if dirName.hasSuffix(".mlmodelc") {
                            // This is a top-level model component
                            scanDirectory(item, component: dirName)
                        } else {
                            // This is a subdirectory
                            scanDirectory(item, component: component)
                        }
                    } else if let fileSize = resourceValues.fileSize {
                        fileCount += 1
                        totalSize += Int64(fileSize)
                        
                        // Track size for this component
                        if let comp = component {
                            componentSizes[comp] = (componentSizes[comp] ?? 0) + Int64(fileSize)
                        }
                        
                        // Track largest files
                        if fileSize > 1_000_000 { // Only track files > 1MB
                            let entry = (name: item.lastPathComponent, size: Int64(fileSize))
                            largestFiles.append(entry)
                            
                            // Keep only 5 largest files
                            largestFiles.sort { $0.size > $1.size }
                            if largestFiles.count > 5 {
                                largestFiles.removeLast()
                            }
                        }
                    }
                }
            } catch {
                output += "Error scanning directory \(url.path): \(error)\n"
            }
        }
        
        // Scan the entire model directory
        scanDirectory(modelDir)
        
        // Calculate size percentage
        let sizePercentage = expectedSize > 0 ? Double(totalSize) / Double(expectedSize) * 100.0 : 0
        let isValidSize = sizePercentage >= 95.0
        
        // Format output
        output += "📊 DETAILED SIZE VERIFICATION for model: \(model.name) (\(modelId))\n"
        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += "• Total file count: \(fileCount)\n"
        output += "• Expected size: \(formatFileSize(expectedSize))\n"
        output += "• Actual size: \(formatFileSize(totalSize))\n"
        output += "• Completeness: \(String(format: "%.1f", sizePercentage))% \(isValidSize ? "✅" : "⚠️")\n"
        output += "\n"
        
        // Component breakdown
        if !componentSizes.isEmpty {
            output += "COMPONENT SIZES:\n"
            output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            
            // Sort by size (largest first)
            let sortedComponents = componentSizes.sorted { $0.value > $1.value }
            for (component, size) in sortedComponents {
                let percentage = Double(size) / Double(totalSize) * 100.0
                output += "• \(component): \(formatFileSize(size)) (\(String(format: "%.1f", percentage))%)\n"
            }
            output += "\n"
        }
        
        // Show largest files
        if !largestFiles.isEmpty {
            output += "LARGEST FILES:\n"
            output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            for (index, file) in largestFiles.enumerated() {
                let percentage = Double(file.size) / Double(totalSize) * 100.0
                output += "\(index + 1). \(file.name): \(formatFileSize(file.size)) (\(String(format: "%.1f", percentage))%)\n"
            }
        }
        
        // Check if size is significantly different from expected
        let sizePercentageFloat = expectedSize > 0 ? Float(totalSize) / Float(expectedSize) * 100.0 : 0
        output.append("\n📊 Size verification:")
        output.append("  - Expected size: \(formatFileSize(expectedSize))")
        output.append("  - Actual size: \(formatFileSize(totalSize))")
        output.append("  - Completeness: \(String(format: "%.1f", sizePercentageFloat))%")
        
        // Update model size if it's significantly different from expected
        if abs(sizePercentageFloat - 100.0) > 5.0 {
            // If actual size is more than 5% different from expected
            output.append("📏 Updating model size from \(formatFileSize(expectedSize)) to \(formatFileSize(totalSize))")
            DispatchQueue.main.async {
                model.size = Int(totalSize)
                
                // Update UserDefaults to persist the correct size
                self.updateCustomModelsInUserDefaults()
            }
        }
        
        // Size validation status
        let hasValidSize = sizePercentageFloat >= 95.0
        let sizeStatusEmoji = hasValidSize ? "✅" : "⚠️"
        output.append("  - Size validation: \(sizeStatusEmoji) \(hasValidSize ? "Valid" : "Incomplete")")
        
        return (isValidSize, totalSize, output)
    }

    // Calculate the actual size of a model on disk
    func calculateActualModelSize(modelId: String) -> Int {
        let modelPath = getModelPath(for: modelId)
        
        // Make sure the directory exists
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            return 0
        }
        
        var totalSize = 0
        
        // Helper function to recursively calculate directory size
        func calculateSize(for url: URL) -> Int {
            var dirSize = 0
            
            do {
                // Get all files in directory
                let contents = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: [.fileSizeKey])
                
                // Sum up the size of all files
                for fileURL in contents {
                    if let fileAttributes = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
                       let fileSize = fileAttributes[.size] as? Int {
                        dirSize += fileSize
                    }
                    
                    // If it's a directory, recursively calculate its size
                    var isDir: ObjCBool = false
                    if FileManager.default.fileExists(atPath: fileURL.path, isDirectory: &isDir), isDir.boolValue {
                        dirSize += calculateSize(for: fileURL)
                    }
                }
            } catch {
                print("Error calculating size for \(url.path): \(error)")
            }
            
            return dirSize
        }
        
        // Calculate size of the entire model directory
        totalSize = calculateSize(for: modelPath)
        
        return totalSize
    }

    func verifyModelWithDetails(modelId: String, verbose: Bool = true) -> (isValid: Bool, actualSize: Int, missingFiles: [String], fileSizes: [String: Int], emptyWeightFiles: [String], errorReason: String?) {
        let modelPath = self.getModelPath(for: modelId)
        var fileSizes: [String: Int] = [:]
        var missingFiles: [String] = []
        var emptyWeightFiles: [String] = []
        var hasCriticalFilesMissing = false
        var errorReason: String? = nil

        // Minimum size for weight.bin files (1KB)
        let minimumWeightFileSize = 1024

        // Get the model to check for hasPlaceholders
        if let model = self.getModel(for: modelId) {
            // Check for critical files
            let tokenizerConfigPath = modelPath.appendingPathComponent("tokenizer_config.json")
            let tokenizerConfigExists = FileManager.default.fileExists(atPath: tokenizerConfigPath.path)

            if !tokenizerConfigExists {
                missingFiles.append("tokenizer_config.json")
                hasCriticalFilesMissing = true
                print("⚠️ Critical file missing: tokenizer_config.json")

                // Update model's hasPlaceholders status on main thread
                DispatchQueue.main.async {
                    model.hasPlaceholders = true
                }
            } else {
                // If tokenizer_config.json exists, check its size
                if let attributes = try? FileManager.default.attributesOfItem(atPath: tokenizerConfigPath.path),
                   let fileSize = attributes[.size] as? Int {
                    fileSizes["tokenizer_config.json"] = fileSize
                    if fileSize == 0 {
                        print("⚠️ tokenizer_config.json exists but is empty")
                        missingFiles.append("tokenizer_config.json (empty)")
                        hasCriticalFilesMissing = true

                        // Update model's hasPlaceholders status on main thread
                        DispatchQueue.main.async {
                            model.hasPlaceholders = true
                        }
                    }
                }
            }

            // Check for empty weight.bin files in .mlmodelc directories
            do {
                let contents = try FileManager.default.contentsOfDirectory(at: modelPath, includingPropertiesForKeys: nil)
                let mlmodelcDirs = contents.filter { $0.pathExtension == "mlmodelc" }

                for mlmodelcDir in mlmodelcDirs {
                    let weightBinPath = mlmodelcDir.appendingPathComponent("weights/weight.bin")
                    if FileManager.default.fileExists(atPath: weightBinPath.path) {
                        if let attributes = try? FileManager.default.attributesOfItem(atPath: weightBinPath.path),
                           let fileSize = attributes[.size] as? Int {
                            let fileName = "\(mlmodelcDir.lastPathComponent)/weights/weight.bin"
                            fileSizes[fileName] = fileSize
                            if fileSize < minimumWeightFileSize {
                                emptyWeightFiles.append(fileName)
                                hasCriticalFilesMissing = true
                                print("⚠️ Weight file is empty/corrupt (0 KB): \(fileName)")
                            }
                        }
                    }
                }
            } catch {
                print("Error checking .mlmodelc directories: \(error)")
            }
        }

        // Use existing verification to check if files exist
        let isValid = !hasCriticalFilesMissing && self.verifyModelFiles(modelId: modelId, verbose: verbose)

        // Calculate actual size on disk
        let actualSize = self.calculateActualModelSize(modelId: modelId)

        // Determine error reason
        if !isValid {
            if !emptyWeightFiles.isEmpty {
                errorReason = "Empty weight files detected (\(emptyWeightFiles.count) files) - re-download required"
            } else if !missingFiles.isEmpty {
                errorReason = "Missing files: \(missingFiles.joined(separator: ", "))"
            } else {
                errorReason = "Model verification failed"
            }
        }

        if verbose {
            print("📋 Model verification details for \(modelId):")
            print("  - Valid: \(isValid)")
            print("  - Actual size: \(formatFileSize(actualSize))")
            print("  - Missing files: \(missingFiles.joined(separator: ", "))")
            print("  - Empty weight files: \(emptyWeightFiles.joined(separator: ", "))")
            print("  - Has critical files missing: \(hasCriticalFilesMissing)")
            if let reason = errorReason {
                print("  - Error reason: \(reason)")
            }
        }

        return (isValid: isValid, actualSize: actualSize, missingFiles: missingFiles, fileSizes: fileSizes, emptyWeightFiles: emptyWeightFiles, errorReason: errorReason)
    }
}

extension ModelService {
    public func checkModelConfiguration(at yamlPath: URL) throws -> ModelConfiguration {
        let yamlContent = try String(contentsOf: yamlPath, encoding: .utf8)
        return try ModelConfiguration(from: yamlContent)
    }
}

// Add this extension at the end of the file, outside of any class or struct
extension NSObject {
    func objc_setAssociatedObject(key: String, value: Any?) {
        let keyPointer = UnsafeRawPointer(bitPattern: key.hashValue)!
        ObjectiveC.objc_setAssociatedObject(self, keyPointer, value, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    }
    
    func objc_getAssociatedObject(key: String) -> Any? {
        let keyPointer = UnsafeRawPointer(bitPattern: key.hashValue)!
        return ObjectiveC.objc_getAssociatedObject(self, keyPointer)
    }
}

