// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// ModelManagementView.swift

import SwiftUI
import Foundation
import AnemllCore
import os.signpost
import QuartzCore
#if os(iOS)
import UIKit
#else
import AppKit
#endif

// MARK: - Logging System

/// Centralized logging system to control debug output
class AppLogger {
    // Log categories
    enum Category: String {
        case models = "📚 Models"
        case ui = "🖥️ UI"
        case performance = "⚡ Performance"
        case lifecycle = "🔄 Lifecycle"
        case network = "📡 Network"
    }
    
    // Log levels
    enum Level: Int {
        case error = 0
        case warning = 1
        case info = 2
        case debug = 3
        case verbose = 4
    }
    
    // Configuration
    static var isLoggingEnabled = true
    static var minLogLevel: Level = .info
    
    // Category filters - set to false to disable specific categories
    static var enabledCategories: [Category: Bool] = [
        .models: true,
        .ui: true,
        .performance: true,
        .lifecycle: true,
        .network: true
    ]
    
    // Production mode toggle - set to true to disable most logs in production
    #if DEBUG
    static let isProductionMode = false
    #else
    static let isProductionMode = true
    #endif
    
    // Log a message
    static func log(_ message: String, category: Category, level: Level = .info, file: String = #file, function: String = #function, line: Int = #line) {
        // Skip logging in production mode unless it's an error
        if isProductionMode && level != .error {
            return
        }
        
        // Skip if logging is disabled
        if !isLoggingEnabled {
            return
        }
        
        // Skip if category is disabled
        if enabledCategories[category] == false {
            return
        }
        
        // Skip if log level is below minimum
        if level.rawValue > minLogLevel.rawValue {
            return
        }
        
        // Format log message
        let levelSymbol: String
        switch level {
        case .error:
            levelSymbol = "❌"
        case .warning:
            levelSymbol = "⚠️"
        case .info:
            levelSymbol = "ℹ️"
        case .debug:
            levelSymbol = "🔍"
        case .verbose:
            levelSymbol = "📝"
        }
        
        // Get filename without path
        let fileName = URL(fileURLWithPath: file).lastPathComponent
        
        // Print formatted message
        print("\(category.rawValue) \(levelSymbol) \(message) [\(fileName):\(line)]")
    }
    
    // Convenience methods for each log level
    static func error(_ message: String, category: Category, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, category: category, level: .error, file: file, function: function, line: line)
    }
    
    static func warning(_ message: String, category: Category, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, category: category, level: .warning, file: file, function: function, line: line)
    }
    
    static func info(_ message: String, category: Category, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, category: category, level: .info, file: file, function: function, line: line)
    }
    
    static func debug(_ message: String, category: Category, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, category: category, level: .debug, file: file, function: function, line: line)
    }
    
    static func verbose(_ message: String, category: Category, file: String = #file, function: String = #function, line: Int = #line) {
        log(message, category: category, level: .verbose, file: file, function: function, line: line)
    }
}

// Main thread monitoring tool
class MainThreadWatcher {
    private var displayLink: CADisplayLink?
    private var lastTimestamp: CFTimeInterval = 0
    private var maxDelay: CFTimeInterval = 0
    private var freezeCount = 0
    
    init() {
        displayLink = CADisplayLink(target: self, selector: #selector(displayLinkFired))
        displayLink?.add(to: .main, forMode: .common)
    }
    
    @objc private func displayLinkFired(link: CADisplayLink) {
        if lastTimestamp == 0 {
            lastTimestamp = link.timestamp
            return
        }
        
        let delay = link.timestamp - lastTimestamp - link.duration
        if delay > 0.1 { // 100ms threshold for considering a UI freeze
            freezeCount += 1
            maxDelay = max(maxDelay, delay)
            // print("⚠️ UI FREEZE DETECTED: \(String(format: "%.2f", delay * 1000))ms, count: \(freezeCount)")
            
            // Print stack trace when significant freeze occurs
            if delay > 0.5 { // 500ms is a very noticeable freeze
                print("STACK TRACE:")
                Thread.callStackSymbols.forEach { print($0) }
            }
        }
        
        lastTimestamp = link.timestamp
    }
    
    deinit {
        displayLink?.invalidate()
    }
}

struct ModelManagementView: View {
    @State private var downloadProgress: [String: Double] = [:]
    @State private var isDownloading: [String: Bool] = [:]
    @State private var currentDownloadingFile: [String: String] = [:]
    @State private var showError = false
    @State private var errorMessage = ""
    @State private var showLocationInfo = false
    @State private var hasPermission: Bool = true
    @State private var showActualPath = false
    @State private var showRedownloadConfirmation = false
    @State private var modelToRedownload: Model? = nil
    @State private var activeModelId: String? = nil
    @State private var showAddModelSheet = false
    @State private var customModelURL = ""
    @State private var customModelName = ""
    @State private var customModelDescription = ""
    @State private var isAddingModel = false
    @State private var showDeleteConfirmation = false
    @State private var modelToDelete: Model? = nil
    @State private var isLoading = false
    @State private var successMessage = ""
    @State private var showSuccess = false
    @State private var refreshID = UUID()
    @State private var showModelLoadConfirmation = false
    @State private var selectedModel: Model? = nil
    @Environment(\.dismiss) private var dismiss
    
    // Add a view will appear handler
    @State private var hasAppeared = false
    
    // Add a property to track if keyboard dismissal has been attempted
    @State private var keyboardDismissalAttempted = false
    
    // Add a flag to track when the custom model sheet is active
    @State private var isCustomModelSheetActive = false
    
    // Debugging tools
    private let mainThreadWatcher = MainThreadWatcher()
    private var uiFreezeDuration: CFTimeInterval = 0
    private let signpostID = OSSignpostID(log: .default)
    private let signposter = OSSignposter()
    
    let modelService = ModelService.shared
    let inferenceService = InferenceService.shared
    
    // Add a computed property for models list
    private var modelsList: [Model] {
        return modelService.getAvailableModels()
    }
    
    // Add a new @State variable to track showing a custom model redownload sheet
    @State private var showCustomRedownloadSheet = false
    
    // Add a new @State variable for download confirmation
    @State private var showDownloadConfirmation = false
    @State private var downloadConfirmationMessage = ""
    
    // Add a new @State variable to track if the selected model has incomplete files
    @State private var hasIncompleteDownload = false
    
    // Add a new state variable to track models with incomplete files
    @State private var modelsWithIncompleteFiles: Set<String> = []
    
    // Add state variables for model info display
    @State private var showModelInfo = false
    @State private var modelForInfo: Model? = nil
    
    // Add state variables for model size verification 
    @State private var modelActualSize: Int = 0
    @State private var modelSizePercentage: Float = 0
    
    // Add a custom initializer to prepare models list before view is created
    init() {
        // The initialization will be handled in onAppear
        // This is just a placeholder to ensure the structure is properly initialized
        // print("ModelManagementView initialized")
    }
    
    // Helper to check if a model is a default model
    private func isDefaultModel(_ model: Model) -> Bool {
        return model.id == "llama-3.2-1b" // Add any other default model IDs here
    }
    
    // Extract the warning banner into a separate component
    private struct RequiredModelBanner: View {
        let modelService: ModelService
        let onDownloadDefaultModel: (Model) -> Void
        @Binding var isCustomModelSheetActive: Bool
        
        var body: some View {
            VStack(alignment: .leading, spacing: 16) {
                HStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.title2)
                        .foregroundColor(.red)
                    
                    VStack(alignment: .leading, spacing: 4) {
                        Text("No Models Available")
                            .font(.title3)
                            .fontWeight(.bold)
                            .foregroundColor(.primary)
                        
                        Text("A language model is required to use the app")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                }
                
                Divider()
                
                VStack(alignment: .leading, spacing: 10) {
                    Text("Recommended Default Model:")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    HStack(spacing: 16) {
                        Image(systemName: "brain.head.profile")
                            .font(.largeTitle)
                            .foregroundColor(.blue)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Llama 3.2 1B")
                                .font(.headline)
                                .foregroundColor(.primary)
                            
                            Text("Small, fast model (~1.6GB)")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                    }
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(12)
                }
                
                // Add download button for Llama 3.2 1B with size info
                Button(action: {
                    // Find the llama-3.2-1b model in the available models
                    if let llamaModel = modelService.getAvailableModels().first(where: { $0.id == "llama-3.2-1b" }) {
                        onDownloadDefaultModel(llamaModel)
                    }
                }) {
                    HStack {
                        Image(systemName: "arrow.down.circle.fill")
                            .font(.title3)
                        Text("DOWNLOAD NOW")
                            .fontWeight(.bold)
                        Spacer()
                        Text("1.6 GB")
                            .font(.subheadline)
                            .foregroundColor(.white.opacity(0.8))
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 14)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .buttonStyle(PlainButtonStyle())
                .padding(.vertical, 8)
                
                HStack {
                    Image(systemName: "wifi")
                        .foregroundColor(.secondary)
                    Text("For the best experience, please connect to WiFi before downloading")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(16)
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.red.opacity(0.3), lineWidth: 2)
            )
            .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
            .padding(.horizontal)
            .padding(.vertical, 8)
            .onAppear {
                // Only dismiss keyboard if custom model sheet is not active
                if !isCustomModelSheetActive {
                    dismissKeyboardForAll()
                }
            }
        }
        
        // More comprehensive keyboard dismissal that targets the entire app
        private func dismissKeyboardForAll() {
            // Use most direct method
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            
            // Force all windows to end editing
            if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
                windowScene.windows.forEach { window in
                    window.endEditing(true)
                }
            }
            
            // Post notification to dismiss any remaining keyboards
            DispatchQueue.main.async {
                NotificationCenter.default.post(name: UIResponder.keyboardWillHideNotification, object: nil)
                NotificationCenter.default.post(name: UIResponder.keyboardDidHideNotification, object: nil)
            }
        }
    }
    
    var body: some View {
        NavigationView {
            mainContentView
                .navigationTitle("Model Management")
                .navigationBarItems(trailing: doneButton)
                .modifier(SafeViewLifecycleModifier(refreshAction: refreshModels))
                .onAppear(perform: handleViewAppear)
                .onDisappear(perform: handleViewDisappear)
        }
        .alert("Load Model", isPresented: $showModelLoadConfirmation) {
            modelLoadConfirmationButtons
        } message: {
            modelLoadConfirmationMessage
        }
        .sheet(isPresented: $isAddingModel) {
            CustomModelSheet(
                customModelURL: $customModelURL,
                customModelName: $customModelName,
                customModelDescription: $customModelDescription,
                isCustomModelSheetActive: $isCustomModelSheetActive,
                onAddCustomModel: addCustomModel,
                isAddingCustomModel: $isAddingModel
            )
        }
        .sheet(isPresented: $showCustomRedownloadSheet) {
            if let model = selectedModel {
                ModelRedownloadSheet(
                    model: model,
                    hasIncompleteDownload: hasIncompleteDownload,
                    actualSize: modelActualSize,
                    sizePercentage: modelSizePercentage,
                    onDownloadMissing: { startDownloadMissing(model) },
                    onDownloadAll: { startDownload(model) },
                    onForceRedownload: { forceRedownloadModel(model) },
                    isPresented: $showCustomRedownloadSheet
                )
            }
        }
        .alert("Download Model", isPresented: $showDownloadConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Download (Resume)") {
                if let model = selectedModel {
                    startDownload(model)
                }
            }
        } message: {
            if let model = selectedModel {
                Text("Do you want to download or resume downloading \(model.name)? Existing files will be preserved.")
            } else {
                Text("Do you want to download this model? Existing files will be preserved.")
            }
        }
        .sheet(isPresented: $showModelInfo) {
            if let model = modelForInfo {
                ModelInfoSheet(model: model, isPresented: $showModelInfo)
            }
        }
        .alert("Delete Model", isPresented: $showDeleteConfirmation) {
            Button("Cancel", role: .cancel) { 
                modelToDelete = nil
            }
            Button("Delete", role: .destructive) {
                if let model = modelToDelete {
                    performModelDeletion(model)
                }
                modelToDelete = nil
            }
        } message: {
            if let model = modelToDelete {
                Text("Are you sure you want to delete '\(model.name)'? This will remove all model files from your device.")
            } else {
                Text("Are you sure you want to delete this model? This will remove all model files from your device.")
            }
        }
    }
    
    // Extract the main content view to avoid complex type checking in body
    private var mainContentView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Check if any models are verified and downloaded correctly
                let verifiedModelsExist = modelsList.contains { model in 
                    model.isDownloaded && !modelsWithIncompleteFiles.contains(model.id)
                }
                
                // If no models are correctly downloaded and verified, show the banner
                if !verifiedModelsExist {
                    // Check for the case where models are marked as downloaded but verification failed
                    let incorrectlyMarkedModels = modelsList.filter { model in
                        model.isDownloaded && modelsWithIncompleteFiles.contains(model.id)
                    }
                    
                    if !incorrectlyMarkedModels.isEmpty {
                        // Show a repair banner specifically for this case
                        VStack(alignment: .leading, spacing: 16) {
                            HStack(spacing: 12) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .font(.title2)
                                    .foregroundColor(.red)
                                
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Model Files Missing")
                                        .font(.title3)
                                        .fontWeight(.bold)
                                        .foregroundColor(.primary)
                                    
                                    Text("Found \(incorrectlyMarkedModels.count) model(s) that need to be repaired")
                                        .font(.subheadline)
                                        .foregroundColor(.secondary)
                                }
                                
                                Spacer()
                            }
                            
                            Divider()
                            
                            // List the models that need repair
                            ForEach(incorrectlyMarkedModels) { model in
                                HStack {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(model.name)
                                            .font(.headline)
                                        Text("Model files are missing or corrupted")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    
                                    Spacer()
                                    
                                    Button(action: {
                                        startDownload(model)
                                    }) {
                                        HStack {
                                            Image(systemName: "arrow.triangle.2.circlepath")
                                            Text("Repair")
                                        }
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 8)
                                        .background(Color.orange)
                                        .foregroundColor(.white)
                                        .cornerRadius(8)
                                    }
                                }
                                .padding()
                                .background(Color.orange.opacity(0.1))
                                .cornerRadius(8)
                            }
                            
                            Text("Please repair or redownload the models to use the app")
                                .font(.callout)
                                .foregroundColor(.secondary)
                                .padding(.top, 8)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(16)
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.red.opacity(0.3), lineWidth: 2)
                        )
                        .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                        .zIndex(100)
                    } else {
                        // Standard banner for when no models are downloaded at all
                        RequiredModelBanner(
                            modelService: modelService,
                            onDownloadDefaultModel: startDownload,
                            isCustomModelSheetActive: $isCustomModelSheetActive
                        )
                        .zIndex(100) // Ensure it's visually on top
                        .transition(.opacity) // Add a nice transition
                        .animation(.easeInOut, value: modelsList.contains(where: { $0.isDownloaded }))
                    }
                }
                
                // Active model section
                ActiveModelSection(
                    modelService: modelService,
                    inferenceService: inferenceService,
                    onLoadModel: loadActiveModel
                )
                
                Divider()
                
                // Available models section with clear heading
                VStack(alignment: .leading, spacing: 8) {
                    if !modelsList.contains(where: { $0.isDownloaded }) {
                        Text("Available Models (Not Downloaded)")
                            .font(.headline)
                            .foregroundColor(.secondary)
                            .padding(.top, 4)
                    }
                    
                    AvailableModelsSection(
                        models: modelsList,
                        onDownload: downloadModel,
                        onDelete: deleteModel,
                        onSelect: selectModel,
                        onLoad: loadModel,
                        onShowInfo: showModelInfo,
                        onCancelDownload: cancelDownload,
                        formatFileSize: formatFileSize,
                        modelService: modelService,
                        isDownloading: $isDownloading,
                        downloadProgress: $downloadProgress,
                        currentDownloadingFile: $currentDownloadingFile,
                        modelsWithIncompleteFiles: modelsWithIncompleteFiles,
                        modelErrors: modelErrors
                    )
                }
                
                Divider()
                
                // Custom model section
                CustomModelSection(
                    customModelURL: $customModelURL,
                    customModelName: $customModelName,
                    isAddingCustomModel: $isAddingModel,
                    isCustomModelSheetActive: $isCustomModelSheetActive,
                    onAddCustomModel: addCustomModel
                )
            }
            .padding()
            .id(refreshID)
        }
        .onTapGesture {
            if !isCustomModelSheetActive {
                let state = signposter.beginInterval("TapGesture", id: signpostID)
                dismissKeyboard()
                signposter.endInterval("TapGesture", state)
            }
        }
        .simultaneousGesture(DragGesture().onChanged { _ in
            if !isCustomModelSheetActive {
                let state = signposter.beginInterval("DragGesture", id: signpostID)
                dismissKeyboard()
                signposter.endInterval("DragGesture", state)
            }
        })
        .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardWillShowNotification)) { _ in
            if !isCustomModelSheetActive {
                let state = signposter.beginInterval("KeyboardWillShow", id: signpostID)
                dismissKeyboard()
                signposter.endInterval("KeyboardWillShow", state)
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardDidShowNotification)) { _ in
            if !isCustomModelSheetActive {
                let state = signposter.beginInterval("KeyboardDidShow", id: signpostID)
                dismissKeyboard()
                signposter.endInterval("KeyboardDidShow", state)
            }
        }
    }
    
    // Confirmation buttons for model loading
    private var modelLoadConfirmationButtons: some View {
        Group {
            Button("No", role: .cancel) { 
                // Just keep the selection but don't load
                if let model = selectedModel {
                    // Make sure the model is still selected, but not marked as loaded
                    modelService.selectModel(model)
                    
                    // Reset any existing loading state to ensure UI is consistent
                    if inferenceService.isModelLoaded {
                        // Only unload if a different model was loaded
                        if inferenceService.currentModel != model.id {
                            inferenceService.unloadModel()
                        }
                    }
                    
                    // Refresh models to update UI state
                    refreshModels(fullRefresh: true)
                    
                    // Show success message
                    successMessage = "Model '\(model.name)' selected. Use the 'Load' button to load it into memory."
                    showSuccess = true
                }
                selectedModel = nil
                
                // Dismiss keyboard if not in custom model sheet
                if !isCustomModelSheetActive {
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                }
            }
            
            Button("Yes", role: .none) {
                if let model = selectedModel {
                    loadSelectedModel(model)
                    selectedModel = nil
                }
            }
        }
    }
    
    // Message for model loading confirmation
    private var modelLoadConfirmationMessage: some View {
        Group {
            if let model = selectedModel {
                Text("Model \"\(model.name)\" is selected. Do you want to load it into memory now?")
            } else {
                Text("Do you want to load this model into memory now?")
            }
        }
    }
    
    // Helper function to handle view appearing
    private func handleViewAppear() {
        // print("🔍 ModelManagementView appeared - Starting performance monitoring")
        
        // Performance tracking
        os_signpost(.event, log: .default, name: "ModelManagement-ViewAppeared")
        
        print("DEBUG: ModelManagementView appeared")
        
        // Use proper keyboard dismissal
        dismissKeyboard()
        
        // Check for models with zero files or completely missing files
        Task {
            // Get models that are marked as downloaded
            let downloadedModels = modelsList.filter { $0.isDownloaded }
            
            // Create a temporary set to avoid UI flickering
            var incompleteModels = Set<String>()
            var completelyMissingModels = Set<String>()
            
            // Map each one to check if it has incomplete files
            for model in downloadedModels {
                // Use the detailed verification with verbose off to reduce logging
                let verificationDetails = modelService.verifyModelWithDetails(modelId: model.id, verbose: false)
                let isValid = verificationDetails.isValid
                let actualSize = verificationDetails.actualSize
                let expectedSize = model.size
                let sizePercentage = Float(actualSize) / Float(expectedSize) * 100.0
                
                if actualSize == 0 || (!isValid && sizePercentage < 5.0) {
                    // Model is completely missing (zero bytes or less than 5% of expected size)
                    print("🚨 Model \(model.id) is completely missing - valid: \(isValid), size: \(formatFileSize(actualSize))/\(formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)")
                    completelyMissingModels.insert(model.id)
                    
                    // Mark it as not downloaded
                    await MainActor.run {
                        model.isDownloaded = false
                    }
                } else if !isValid || sizePercentage < 95.0 {
                    // If files are missing or size is significantly less than expected
                    print("⚠️ Model \(model.id) is incomplete - valid: \(isValid), size: \(formatFileSize(actualSize))/\(formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)")
                    incompleteModels.insert(model.id)
                }
            }
            
            // Update UI status
            await MainActor.run {
                // Update the state variable for incomplete but present models
                self.modelsWithIncompleteFiles = incompleteModels
                
                // Force refresh of UI to show downloads needed
                if !completelyMissingModels.isEmpty {
                    print("🔄 Detected \(completelyMissingModels.count) models marked as downloaded but completely missing. Fixed status.")
                    self.refreshModels(fullRefresh: true)
                }
            }
        }
        
        // Delay to ensure view is fully presented
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            self.dismissKeyboard()
            
            // If this is the first appearance, load models on background thread
            if !self.hasAppeared {
                self.hasAppeared = true
                
                // Load models asynchronously to prevent UI freezes
                Task {
                    let state = self.signposter.beginInterval("LoadModels", id: self.signpostID)
                    
                    // print("⏳ Loading models asynchronously")
                    
                    // Perform the model loading on a background thread
                    await Task.detached(priority: .userInitiated) {
                        // Get the list of available models
                        _ = ModelService.shared.getAvailableModels()
                        // print("📱 Loaded \(models.count) models from ModelService")
                    }.value
                    
                    // Update UI on main thread
                    await MainActor.run {
                        self.refreshModels(fullRefresh: true)
                    }
                    
                    
                    self.signposter.endInterval("LoadModels", state)
                }
            }
        }
        
        setupObservers()
        refreshModels(fullRefresh: false)
        checkIfModelIsLoadedForInference()
        
        // Check for models with incomplete files
        Task {
            // Find models that are marked as downloaded
            let downloadedModels = modelService.getAvailableModels().filter { $0.isDownloaded }
            
            // Create a temporary set to avoid UI flickering
            var incompleteModels = Set<String>()
            
            // Check each downloaded model for incomplete files
            for model in downloadedModels {
                // Use the detailed verification method
                let verificationDetails = modelService.verifyModelWithDetails(modelId: model.id, verbose: false)
                let isValid = verificationDetails.isValid
                let actualSize = verificationDetails.actualSize
                let expectedSize = model.size
                let sizePercentage = Float(actualSize) / Float(expectedSize) * 100.0
                
                if !isValid || sizePercentage < 95.0 {
                    // If files are missing or size is significantly less than expected (less than 95%)
                    AppLogger.warning("Model \(model.name) is incomplete - valid: \(isValid), size: \(formatFileSize(actualSize))/\(formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)", category: .models)
                    incompleteModels.insert(model.id)
            } else {
                    AppLogger.info("Model \(model.name) verification successful - size: \(formatFileSize(actualSize))/\(formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)", category: .models)
                }
            }
            
            // Update the set of models with incomplete files
            modelsWithIncompleteFiles = incompleteModels
        }
        
        // Register notification observers
                            NotificationCenter.default.addObserver(
            forName: Notification.Name("DownloadDefaultModel"),
                                object: nil, 
                                queue: .main
        ) { _ in
            // Get the default model (assuming the first model is default)
            if let defaultModel = self.modelService.getAvailableModels().first {
                AppLogger.info("Received notification to download default model: \(defaultModel.name)", category: .models)
                self.downloadModel(defaultModel)
            }
        }
        
        // Observer for downloading a specific model by ID
                            NotificationCenter.default.addObserver(
            forName: Notification.Name("DownloadSelectedModel"),
                                object: nil,
                                queue: .main
        ) { notification in
            guard let modelId = notification.object as? String else { return }
            
            // Find the model with the given ID
            if let modelToDownload = self.modelService.getAvailableModels().first(where: { $0.id == modelId }) {
                AppLogger.info("Received notification to download specific model: \(modelToDownload.name)", category: .models)
                self.downloadModel(modelToDownload)
            }
        }
    }
    
    // Helper function to handle view disappearing
    private func handleViewDisappear() {
        // print("ModelManagementView disappeared")
        // Clear any selected model to prevent stale references
        selectedModel = nil
        
        // Cancel any in-progress verification tasks
        Task {
            // Log that we're cleaning up
            print("Cleaning up background tasks when ModelManagementView disappeared")
            // No more need to update UI
            isLoading = false
        }
        
        // Remove notification observers
        NotificationCenter.default.removeObserver(self, name: Notification.Name("DownloadDefaultModel"), object: nil)
        NotificationCenter.default.removeObserver(self, name: Notification.Name("DownloadSelectedModel"), object: nil)
    }
    
    // Load models asynchronously to prevent UI freezes
    private func loadModelsAsync() async {
        let state = signposter.beginInterval("LoadModels", id: signpostID)
        
        AppLogger.info("Loading models asynchronously", category: .performance)
        await MainActor.run {
            isLoading = true
        }
        
        // Perform the model loading on a background thread
        await Task.detached(priority: .userInitiated) {
            // Get the list of available models
            let models = ModelService.shared.getAvailableModels()
            AppLogger.info("Loaded \(models.count) models from ModelService", category: .models)
            
            // Verify model files in background with reduced logging
            for model in models where model.isDownloaded {
                let isValid = ModelService.shared.verifyModelFiles(modelId: model.id)
                AppLogger.info("Model \(model.id) verification: \(isValid ? "✅" : "❌")", category: .models)
            }
        }.value
        
        // Update UI on main thread
        await MainActor.run {
            isLoading = false
            refreshModels(fullRefresh: true)
        }
        
        signposter.endInterval("LoadModels", state)
    }
    
    // MARK: - File Operation Helpers
    
    // Move file operations off the main thread for better UI responsiveness
    private func checkFileExists(at path: URL) async -> Bool {
        return await Task.detached(priority: .background) {
            return FileManager.default.fileExists(atPath: path.path)
        }.value
    }
    
    private func getFileSize(at path: URL) async -> UInt64? {
        return await Task.detached(priority: .background) {
            do {
                let attributes = try FileManager.default.attributesOfItem(atPath: path.path)
                return attributes[FileAttributeKey.size] as? UInt64
            } catch {
                AppLogger.error("Failed to get file size: \(error)", category: .network)
                return nil
            }
        }.value
    }
    
    private func readFileContents(at path: URL) async -> String? {
        return await Task.detached(priority: .background) {
            do {
                return try String(contentsOf: path, encoding: .utf8)
            } catch {
                AppLogger.error("Failed to read file contents: \(error)", category: .network)
                return nil
            }
        }.value
    }
    
    private func createDirectory(at path: URL) async -> Bool {
        return await Task.detached(priority: .background) {
            do {
                try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true, attributes: nil)
                return true
            } catch {
                AppLogger.error("Failed to create directory: \(error)", category: .network)
                return false
            }
        }.value
    }
    
    // Optimized model refresh with performance tracking
    private func refreshModels(fullRefresh: Bool = true) {
        Task {
            // Get models that are marked as downloaded
            let downloadedModels = modelsList.filter { $0.isDownloaded }
            
            // Create a temporary set to avoid UI flickering
            var incompleteModels = Set<String>()
            var completelyMissingModels = Set<String>()
            var modelErrors: [String: String] = [:]  // Track specific errors for each model
            
            // Map each one to check if it has incomplete files
            for model in downloadedModels {
                // Use the detailed verification with verbose off to reduce logging
                let verificationDetails = modelService.verifyModelWithDetails(modelId: model.id, verbose: false)
                let isValid = verificationDetails.isValid
                let actualSize = verificationDetails.actualSize
                let expectedSize = model.size
                let sizePercentage = Float(actualSize) / Float(expectedSize) * 100.0
                
                // Check for specific LUT-related errors
                let missingFiles = verificationDetails.missingFiles
                let lutErrors = missingFiles.filter { $0.contains("_lut") }
                if !lutErrors.isEmpty {
                    let errorMsg = "Missing LUT files: \(lutErrors.joined(separator: ", "))"
                    modelErrors[model.id] = errorMsg
                    print("🚨 \(errorMsg)")
                    incompleteModels.insert(model.id)
                }
                
                if actualSize == 0 || (!isValid && sizePercentage < 5.0) {
                    // Model is completely missing
                    print("🚨 Model \(model.id) is completely missing - valid: \(isValid), size: \(formatFileSize(actualSize))/\(formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)")
                    completelyMissingModels.insert(model.id)
                    
                    // Mark it as not downloaded
                    await MainActor.run {
                        model.isDownloaded = false
                    }
                } else if !isValid || sizePercentage < 95.0 {
                    // If files are missing or size is significantly less than expected
                    print("⚠️ Model \(model.id) is incomplete - valid: \(isValid), size: \(formatFileSize(actualSize))/\(formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)")
                    incompleteModels.insert(model.id)
                    
                    // Add size error if no LUT error already exists
                    if modelErrors[model.id] == nil {
                        modelErrors[model.id] = "Incomplete download: \(String(format: "%.1f", sizePercentage))% of expected size"
                    }
                }
            }
            
            // Update UI status
            await MainActor.run {
                self.modelsWithIncompleteFiles = incompleteModels
                // Store error messages for display
                self.modelErrors = modelErrors
                
                // If this was a full refresh, update the refresh token
                if fullRefresh {
                    self.refreshID = UUID()
                }
            }
        }
    }
    
    // Timer management - use a state property instead of direct timer reference
    // Helper function to dismiss keyboard in multiple ways
    private func dismissKeyboard() {
        let state = signposter.beginInterval("KeyboardDismiss", id: signpostID)
        
        // Skip keyboard dismissal if the custom model sheet is active
        if isCustomModelSheetActive {
            // print("DEBUG: Skipping keyboard dismissal due to active custom model sheet")
            signposter.endInterval("KeyboardDismiss", state)
            return
        }
        
        // Only attempt keyboard dismissal if we haven't already tried recently
        // This prevents excessive dismissal attempts
        if !keyboardDismissalAttempted {
            // print("DEBUG: Dismissing keyboard")
            keyboardDismissalAttempted = true
            
            // Use a single method rather than multiple competing methods
            #if os(iOS)
            DispatchQueue.main.async {
                if !self.isCustomModelSheetActive {
                    // Primary method: Give focus to a non-text view to dismiss keyboard gracefully
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                
                    // Reset the flag after a delay
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        self.keyboardDismissalAttempted = false
                    }
                }
            }
            #endif
        } else {
            // print("DEBUG: Keyboard dismissal already attempted")
        }
        
        signposter.endInterval("KeyboardDismiss", state)
    }
    
    private func dismissKeyboardAndCleanup() {
        let state = signposter.beginInterval("KeyboardCleanup", id: signpostID)
        
        dismissKeyboard()
        
        #if os(iOS)
        // Clean up any orphaned keyboard controllers and clear input fields
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
            windowScene.windows.forEach { window in
                window.subviews.forEach { view in
                    view.endEditing(true)
                }
            }
        }
        
        // Force keyboard dismissal with notification
        NotificationCenter.default.post(name: UIResponder.keyboardWillHideNotification, object: nil)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            NotificationCenter.default.post(name: UIResponder.keyboardDidHideNotification, object: nil)
        }
        #endif
        
        signposter.endInterval("KeyboardCleanup", state)
    }
    
    // MARK: - Model Action Handlers
    
    // Function to start downloading a model
    private func startDownload(_ model: Model) {
        print("⬇️ Starting download for model: \(model.id)")
        
        // Mark model as downloading
                        isDownloading[model.id] = true
                        
        // Reset progress 
        downloadProgress[model.id] = 0.01  // Start with non-zero progress to make it visible
        
        // Initiate model download
        Task {
            // Reset download progress in case of restart
            self.downloadProgress[model.id] = 0.01
            self.currentDownloadingFile[model.id] = "Preparing download..."
            
            // Force initial UI update
            self.refreshID = UUID()
            
            modelService.downloadModel(modelId: model.id) { file, progress in
                // Update progress on main thread
                DispatchQueue.main.async {
                    // Always use a non-zero progress value
                    let visibleProgress = max(0.01, progress)
                    
                    // Debug progress updates
                    print("🔄 UI Progress update: \(model.id) - \(file) - \(Int(visibleProgress * 100))%")
                    
                    // Update local state
                    self.downloadProgress[model.id] = visibleProgress
                    self.currentDownloadingFile[model.id] = file
                    
                    // Always force UI refresh for progress updates
                    self.refreshID = UUID()
                }
            } completion: { success in
                // Update UI on completion
                DispatchQueue.main.async {
                    print("✅ Download completed for model: \(model.id), success: \(success)")
                    self.isDownloading[model.id] = false
                    
                    // Final update to progress
                    if success {
                        self.downloadProgress[model.id] = 1.0
                        self.currentDownloadingFile[model.id] = "Download complete!"
                        
                        // Add verification after download completes
                        Task {
                            // Perform verification in background
                            print("🔍 Verifying model: \(model.id)")
                            let verificationDetails = self.modelService.verifyModelWithDetails(modelId: model.id, verbose: false)
                            let isValid = verificationDetails.isValid
                            let actualSize = verificationDetails.actualSize
                            let expectedSize = model.size
                            let sizePercentage = Float(actualSize) / Float(expectedSize) * 100.0
                            
                            // Log verification results
                            if !isValid || sizePercentage < 95.0 {
                                // If files are missing or size is significantly less than expected
                                print("⚠️ Downloaded model \(model.id) verification failed - valid: \(isValid), size: \(self.formatFileSize(actualSize))/\(self.formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)")
                                
                                // Update incomplete models collection
                                _ = await MainActor.run {
                                    self.modelsWithIncompleteFiles.insert(model.id)
                                }
                            } else {
                                print("✅ Downloaded model \(model.id) verification successful - size: \(self.formatFileSize(actualSize))/\(self.formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)")
                                
                                // Remove from incomplete models if it was there
                                _ = await MainActor.run {
                                    self.modelsWithIncompleteFiles.remove(model.id)
                                }
                            }
                            
                            // Perform a full UI refresh after verification
                            await MainActor.run {
                                self.refreshModels(fullRefresh: true)
                            }
                        }
                    } else {
                        self.downloadProgress[model.id] = 0.0
                        self.currentDownloadingFile[model.id] = "Download failed"
                    }
                    
                    // Force immediate UI refresh to update download status
                    self.refreshID = UUID()
                    
                    // Show success message if download was successful
                    if success {
                        self.successMessage = "Model '\(model.name)' downloaded successfully."
                        self.showSuccess = true
                    }
                }
            }
        }
    }
    
    // Function to load a model for active use
    private func loadActiveModel(_ model: Model) {
        // print("Loading model for active use: \(model.id)")
        activeModelId = model.id
        
        // Call inference service to load the model
        Task {
            do {
                try await inferenceService.loadModel(modelId: model.id, from: modelService.getModelPath(for: model.id))
                
                // Update UI on completion
                await MainActor.run {
                    successMessage = "Model '\(model.name)' loaded successfully."
                    showSuccess = true
                }
            } catch {
                // Handle load error
                await MainActor.run {
                    errorMessage = "Failed to load model: \(error.localizedDescription)"
                    showError = true
                }
            }
        }
    }
    
    // Function to handle user selection of a model
    private func selectModel(_ model: Model) {
        // print("Selected model: \(model.id)")
        
        // First check if there's an active model loading process
        if inferenceService.isLoadingModel {
            // Cancel any ongoing model loading with proper reason
            inferenceService.cancelModelLoading(reason: .startingNewModel)
            
            // Add a small delay to ensure cancellation completes
            Task {
                try? await Task.sleep(nanoseconds: 300_000_000) // 0.3 seconds delay
                
                // Show confirmation dialog
                await MainActor.run {
                    selectedModel = model
                    showModelLoadConfirmation = true
                }
            }
        } else {
            // No active loading, proceed with selection and show confirmation dialog
            selectedModel = model
            showModelLoadConfirmation = true
        }
    }
    
    // Helper function to select model after ensuring no active loading
    private func selectModelAfterCancellation(_ model: Model) async {
        // Check if there's a currently loaded model that's different from the one we're selecting
        if let loadedModel = inferenceService.getCurrentlyLoadedModel(),
           loadedModel.id != model.id {
            // Unload the currently loaded model - no need for await since it's not async
            inferenceService.unloadModel()
            
            // Now set the new selected model
            modelService.selectModel(model)
            
            // Update UI on main thread after unloading
            await MainActor.run {
                // Refresh the UI immediately to show the selection
                refreshModels(fullRefresh: true)
                
                // Show success message about selection and unloading
                successMessage = "Model '\(model.name)' selected and previous model unloaded. Use the 'Load' button to load it into memory."
                showSuccess = true
            }
                                } else {
            // No model is currently loaded, or the selected model is already loaded
            modelService.selectModel(model)
            
            // Update UI on main thread
            await MainActor.run {
                // Refresh the UI immediately to show the selection
                refreshModels(fullRefresh: true)
                
                // Show success message about selection
                successMessage = "Model '\(model.name)' selected. Use the 'Load' button to load it into memory."
                showSuccess = true
            }
        }
    }
    
    // Function to load a model (without selecting it first)
    private func loadModel(_ model: Model) {
        // print("Loading model: \(model.id)")
        
        // Check if there's already a model loading in progress
        if inferenceService.isLoadingModel {
            // Cancel the current loading process
            inferenceService.cancelModelLoading(reason: .startingNewModel)
            
            // Wait a moment to ensure cancellation is complete
            Task {
                try? await Task.sleep(nanoseconds: 300_000_000) // 0.3 seconds
                
                // Now show the load confirmation
                await MainActor.run {
                    selectedModel = model
                    showModelLoadConfirmation = true
                }
            }
        } else {
            // No loading in progress, proceed normally
            selectedModel = model
            showModelLoadConfirmation = true
        }
    }
    
    // Function to load the selected model
    private func loadSelectedModel(_ model: Model) {
        // print("Loading selected model: \(model.id)")
        
        // First make sure the model is selected
        modelService.selectModel(model)
        
        // Check if there's already a model loading in progress
        if inferenceService.isLoadingModel {
            // Cancel the current loading process
            inferenceService.cancelModelLoading(reason: .startingNewModel)
            
            // Wait a moment to ensure cancellation is complete
            Task {
                try? await Task.sleep(nanoseconds: 300_000_000) // 0.3 seconds
                
                // Now load the selected model
                loadActiveModel(model)
                
                // Dismiss the model management view
                await MainActor.run {
                    dismiss()
                }
            }
        } else {
            // No loading in progress, proceed normally
            loadActiveModel(model)
            
            // Dismiss the model management view
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                self.dismiss()
            }
        }
    }
    
    // Function to cancel an in-progress download
    private func cancelDownload(_ model: Model) {
        // print("Cancelling download for model: \(model.id)")
        
        Task {
            // Remove await since cancelDownload isn't async
            modelService.cancelDownload(modelId: model.id)
            
            // Update UI
            await MainActor.run {
                isDownloading[model.id] = false
                downloadProgress[model.id] = 0.0
                currentDownloadingFile[model.id] = ""
            }
        }
    }
    
    // Function to delete a downloaded model
    private func deleteModel(_ model: Model) {
        // print("Preparing to delete model: \(model.id)")
        modelToDelete = model
        showDeleteConfirmation = true
    }
    
    // Function to handle actual model deletion
    private func performModelDeletion(_ model: Model) {
        // First check if this is the default LLAMA 1B model or a custom model
        let isDefaultLlama = model.id == "llama-3.2-1b"
        
        // Get the model path to delete files
        let modelPath = modelService.getModelPath(for: model.id)
        
        // Show a loading indicator
        isLoading = true
        
        Task {
            do {
                // Delete the model files from disk
                if FileManager.default.fileExists(atPath: modelPath.path) {
                    try FileManager.default.removeItem(at: modelPath)
                    print("Successfully deleted model files for: \(model.id)")
                }
                
                // For custom models, update Models.json to remove the model
                if !isDefaultLlama {
                    // Remove the custom model from the service
                    modelService.removeCustomModel(modelId: model.id)
                    print("Removed custom model from available models: \(model.id)")
                } else {
                    // For default models, just mark as not downloaded
                    model.isDownloaded = false
                    print("Marked default model as not downloaded: \(model.id)")
                }
                
                // Refresh the UI to show updated model status
                await MainActor.run {
                    successMessage = "Model '\(model.name)' has been deleted."
                    showSuccess = true
                    isLoading = false
                    
                    // Refresh the view
                    refreshModels(fullRefresh: true)
                }
            } catch {
                // Handle deletion errors
                await MainActor.run {
                    errorMessage = "Failed to delete model: \(error.localizedDescription)"
                    showError = true
                    isLoading = false
                }
            }
        }
    }
    
    // Function to add a custom model
    private func addCustomModel() {
        // print("Adding custom model from URL: \(customModelURL)")
        
        // Validate URL
        guard !customModelURL.isEmpty else {
            errorMessage = "Model URL cannot be empty."
            showError = true
            return
        }
        
        // Prepare model name
        let finalModelName = customModelName.isEmpty ? 
            URL(string: customModelURL)?.lastPathComponent ?? "Custom Model" : 
            customModelName
        
        isAddingModel = false
        
        // Add the custom model
        Task {
            do {
                var success = false
                var errorMsg: String? = nil
                
                // Call the ModelService method with completion handler
                await withCheckedContinuation { continuation in
                    modelService.addCustomModel(
                        name: finalModelName,
                        description: customModelDescription,
                        downloadURL: customModelURL
                    ) { didSucceed, error in
                        success = didSucceed
                        errorMsg = error
                        continuation.resume()
                    }
                }
                
                // Check if successful
                if success {
                    // Update UI on completion
                    await MainActor.run {
                        successMessage = "Custom model '\(finalModelName)' added successfully."
                        showSuccess = true
                        
                        // Reset fields
                        customModelURL = ""
                        customModelName = ""
                        customModelDescription = ""
                        
                        // Refresh the models list
                        refreshModels(fullRefresh: true)
                    }
                } else {
                    throw NSError(domain: "ModelService", code: 1, userInfo: [NSLocalizedDescriptionKey: errorMsg ?? "Unknown error"])
                }
            } catch {
                // Handle error
                await MainActor.run {
                    errorMessage = "Failed to add custom model: \(error.localizedDescription)"
                    showError = true
                }
            }
        }
    }
    
    // Function to download missing files
    private func startDownloadMissing(_ model: Model) {
        print("Starting download of missing files for model: \(model.id)")
        
        // Mark model as downloading
        isDownloading[model.id] = true
        
        // Reset progress
        downloadProgress[model.id] = 0.0
        
        // Initiate download of missing files
        modelService.downloadMissingFiles(model: model) { file, progress in
            // Update progress on main thread
            DispatchQueue.main.async {
                self.downloadProgress[model.id] = progress
                self.currentDownloadingFile[model.id] = file
            }
        }
        
        // Handle download completion and verification
        Task {
            // Wait for the download to complete (this is a placeholder for proper download completion detection)
            // Ideally this would be handled with a proper async/await pattern or callback from downloadMissingFiles
            try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds to allow download to complete
            
            // Update UI to show download complete
            await MainActor.run {
                self.isDownloading[model.id] = false
                self.downloadProgress[model.id] = 1.0
                self.currentDownloadingFile[model.id] = "Verifying..."
                
                // Show success message
                self.successMessage = "Missing files for '\(model.name)' downloaded successfully."
                self.showSuccess = true
            }
            
            // Perform verification
            print("🔍 Verifying after downloading missing files for model: \(model.id)")
            let verificationDetails = self.modelService.verifyModelWithDetails(modelId: model.id, verbose: false)
            let isValid = verificationDetails.isValid
            let actualSize = verificationDetails.actualSize
            let expectedSize = model.size
            let sizePercentage = Float(actualSize) / Float(expectedSize) * 100.0
            
            // Log verification results
            if !isValid || sizePercentage < 95.0 {
                // If files are missing or size is significantly less than expected
                print("⚠️ Model \(model.id) still has issues after repair - valid: \(isValid), size: \(self.formatFileSize(actualSize))/\(self.formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)")
                
                // Update incomplete models collection
                _ = await MainActor.run {
                    self.modelsWithIncompleteFiles.insert(model.id)
                }
            } else {
                print("✅ Model \(model.id) repair successful - size: \(self.formatFileSize(actualSize))/\(self.formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)")
                
                // Remove from incomplete models if it was fixed
                _ = await MainActor.run {
                    self.modelsWithIncompleteFiles.remove(model.id)
                }
            }
            
            // Update UI with completed status
            await MainActor.run {
                self.currentDownloadingFile[model.id] = "Verification complete"
                self.refreshModels(fullRefresh: true)
            }
        }
    }
    
    // Function to force a complete redownload
    private func forceRedownloadModel(_ model: Model) {
        print("Force redownloading model: \(model.id)")
        
        // Mark model as downloading
        isDownloading[model.id] = true
        
        // Reset progress
        downloadProgress[model.id] = 0.0
        
        // Initiate full redownload
        modelService.forceRedownload(model: model) { file, progress in
            // Update progress on main thread
            DispatchQueue.main.async {
                self.downloadProgress[model.id] = progress
                self.currentDownloadingFile[model.id] = file
            }
        }
        
        // Handle download completion and verification 
        Task {
            // Wait for the download to complete (this is a placeholder for proper download completion detection)
            // Ideally this would be handled with a proper async/await pattern or callback
            try? await Task.sleep(nanoseconds: 5_000_000_000) // 5 seconds to allow for a full redownload
            
            // Update UI to show download complete
            await MainActor.run {
                self.isDownloading[model.id] = false
                self.downloadProgress[model.id] = 1.0
                self.currentDownloadingFile[model.id] = "Verifying..."
                
                // Show success message
                self.successMessage = "Model '\(model.name)' redownloaded successfully."
                self.showSuccess = true
            }
            
            // Perform verification after full redownload
            print("🔍 Verifying after complete redownload of model: \(model.id)")
            let verificationDetails = self.modelService.verifyModelWithDetails(modelId: model.id, verbose: false)
            let isValid = verificationDetails.isValid
            let actualSize = verificationDetails.actualSize
            let expectedSize = model.size
            let sizePercentage = Float(actualSize) / Float(expectedSize) * 100.0
            
            // Log verification results
            if !isValid || sizePercentage < 95.0 {
                // If files are missing or size is significantly less than expected
                print("⚠️ Model \(model.id) still has issues after full redownload - valid: \(isValid), size: \(self.formatFileSize(actualSize))/\(self.formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)")
                
                // Update incomplete models collection
                _ = await MainActor.run {
                    self.modelsWithIncompleteFiles.insert(model.id)
                }
            } else {
                print("✅ Model \(model.id) full redownload successful - size: \(self.formatFileSize(actualSize))/\(self.formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)")
                
                // Remove from incomplete models as it should be fixed after complete redownload
                _ = await MainActor.run {
                    self.modelsWithIncompleteFiles.remove(model.id)
                }
            }
            
            // Update UI with completed status
            await MainActor.run {
                self.currentDownloadingFile[model.id] = "Verification complete"
                self.refreshModels(fullRefresh: true)
            }
        }
    }
    
    // Function to initiate model download
    private func downloadModel(_ model: Model) {
        // print("Initiating download for model: \(model.id)")
        selectedModel = model
        
        if model.isDownloaded {
            // Check if the model needs verification
            let verificationDetails = modelService.verifyModelWithDetails(modelId: model.id, verbose: false)
            let hasIncompleteFiles = !verificationDetails.isValid
            let actualSize = verificationDetails.actualSize
            let expectedSize = model.size
            let sizePercentage = Float(actualSize) / Float(expectedSize) * 100.0
            
            // Store verification details for the model info view
            self.hasIncompleteDownload = hasIncompleteFiles
            self.modelActualSize = actualSize
            self.modelSizePercentage = sizePercentage
            
            // If this model has already been detected as having incomplete files,
            // show the redownload sheet with appropriate flags set
            self.showCustomRedownloadSheet = true
            
            if hasIncompleteFiles {
                print("⚠️ Model \(model.id) has incomplete files - showing verification options")
                print("📊 Size verification: \(formatFileSize(actualSize)) / \(formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)")
            } else {
                print("✅ Model \(model.id) appears complete - showing redownload options")
                print("📊 Size verification: \(formatFileSize(actualSize)) / \(formatFileSize(expectedSize)) (\(String(format: "%.1f", sizePercentage))%)")
            }
        } else {
            // Show regular download confirmation for not-yet-downloaded models
            downloadConfirmationMessage = "Are you sure you want to download \(model.name)? The model is \(formatFileSize(model.size))."
            showDownloadConfirmation = true
        }
    }
    
    // Function to show model info
    private func showModelInfo(_ model: Model) {
        // print("Showing info for model: \(model.id)")
        modelForInfo = model
        showModelInfo = true
    }
    
    // MARK: - Lifecycle Management
    
    // Helper functions for lifecycle management
    private func setupObservers() {
        // Set up model-specific observers
        NotificationCenter.default.addObserver(
            forName: Notification.Name("ModelLoadingFailed"),
            object: nil,
            queue: .main
        ) { notification in
            // Access userInfo safely
            if let userInfo = notification.userInfo,
               let errorMsg = userInfo["error"] as? String {
                // Keep error prints for debugging
                print("⚠️ Model loading failed: \(errorMsg)")
            } else {
                print("⚠️ Model loading failed without details")
            }
        }
        
        // Add observer for model loading progress changes
        NotificationCenter.default.addObserver(
            forName: Notification.Name("ModelLoadingProgressUpdated"),
            object: nil,
            queue: .main
        ) { [self] _ in
            Task { @MainActor in
                // Only update the UI without verification during active model loading
                if self.inferenceService.isLoadingModel {
                    // Just trigger UI refresh without running verification
                    self.refreshID = UUID()
                } else {
                    // Regular update with verification when not loading a model
                    self.refreshModels(fullRefresh: false)
                }
            }
        }
        
        // Add observer for model loading state changes
        NotificationCenter.default.addObserver(
            forName: Notification.Name("ModelLoadingStateChanged"),
            object: nil,
            queue: .main
        ) { [self] _ in
            // Update the UI when loading state changes
            refreshModels(fullRefresh: true)
        }
        
        // print("🔔 ModelManagementView observers set up")
    }
    
    private func checkIfModelIsLoadedForInference() {
        // Check if there's an active model
        if let activeModel = inferenceService.getCurrentlyLoadedModel() {
            activeModelId = activeModel.id
            // print("📱 Active model detected: \(activeModel.name)")
        } else {
            activeModelId = nil
            // print("📱 No active model detected")
        }
    }
    
    // Modifier to handle safe view lifecycle
    private struct SafeViewLifecycleModifier: ViewModifier {
        let refreshAction: (Bool) -> Void
        
        func body(content: Content) -> some View {
            content
                .onAppear {
                    // Perform a lightweight refresh when view appears
                    refreshAction(false)
                }
                .onDisappear {
                    // Clean up any resources when view disappears
                    // print("SafeViewLifecycleModifier: View disappeared")
                }
        }
    }
    
    // Helper function to format source URLs for display
    private func formatSourceURL(_ url: String) -> String {
        // If URL is a filesystem path, mask it
        if url.starts(with: "/") || url.starts(with: "~") || url.contains("SourceRelease") {
            // For local filesystem URLs, use a generic format
            return "${MODEL_ROOT}/models/custom/\(URL(fileURLWithPath: url).lastPathComponent)"
        } else if url.starts(with: "file://") {
            // Extract and mask file URLs
            let path = url.replacingOccurrences(of: "file://", with: "")
            return "${MODEL_ROOT}/models/custom/\(URL(fileURLWithPath: path).lastPathComponent)"
        }
        // For web URLs, return as is
        return url
    }
    
    // Add property to store model errors
    @State private var modelErrors: [String: String] = [:]
}

// MARK: - UI Components

// Component for displaying the currently active model
struct ActiveModelSection: View {
    let modelService: ModelService
    let inferenceService: InferenceService
    let onLoadModel: (Model) -> Void
    
    // Add a state variable for refreshing the view
    @State private var refreshToken = UUID()
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "bolt.fill")
                    .foregroundColor(.yellow)
                Text("Active Model")
                    .font(.headline)
                Spacer()
            }
            
            // First check if there's a model being loaded
            if inferenceService.isLoadingModel {
                HStack(alignment: .center) {
                    VStack(alignment: .leading, spacing: 4) {
                        if let selectedModel = modelService.getSelectedModel() {
                            Text("\(selectedModel.name)")
                                .font(.subheadline)
                                .fontWeight(.medium)
                } else {
                            Text("Loading model...")
                                .font(.subheadline)
                                .fontWeight(.medium)
                        }
                        
                        HStack {
                            Text("Loading: \(Int(inferenceService.loadingProgress * 100))%")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Text(inferenceService.loadingStatus)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .lineLimit(1)
                                .truncationMode(.tail)
                        }
                    }
                    
                    Spacer()
                    
                    HStack(spacing: 12) {
                        // Loading indicator
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
                            .scaleEffect(0.8)
                        
                        // Cancel button
                    Button(action: {
                            inferenceService.cancelModelLoading(reason: .userInitiated)
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.red)
                                .font(.title3)
                        }
                    }
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 12)
                .background(Color(.systemGray6))
                .cornerRadius(8)
                .id(refreshToken) // Use the refreshToken to force view updates
            }
            // Then check if there's a loaded model in the inference engine
            else if let activeModel = inferenceService.getCurrentlyLoadedModel() {
                HStack(alignment: .center) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(activeModel.name)
                            .font(.subheadline)
                            .fontWeight(.medium)
                        Text("Size: \(formatFileSize(activeModel.size))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    Text("Loaded & Active")
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.green.opacity(0.2))
                        .foregroundColor(.green)
                        .cornerRadius(4)
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 12)
                .background(Color(.systemGray6))
                .cornerRadius(8)
            }
            // If no loaded model, check if there's a selected model that's not loaded
            else if let selectedModel = modelService.getSelectedModel() {
                HStack(alignment: .center) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(selectedModel.name)
                            .font(.subheadline)
                            .fontWeight(.medium)
                        Text("Size: \(formatFileSize(selectedModel.size))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    HStack(spacing: 8) {
                        Text("Selected")
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.blue.opacity(0.2))
                            .foregroundColor(.blue)
                            .cornerRadius(4)
                        
                        // Add load button
                        Button(action: {
                            onLoadModel(selectedModel)
                        }) {
                            Text("Load")
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.purple)
                                .foregroundColor(.white)
                                .cornerRadius(4)
                        }
                    }
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 12)
                .background(Color(.systemGray6))
                .cornerRadius(8)
            }
            else {
                Text("No model is currently active")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .padding(.vertical, 8)
            }
        }
        .onAppear {
            // Start a timer to refresh the view when it appears
        }
        .onDisappear {
            // Cancel the timer when the view disappears
        }
    }
    
    
    
    private func formatFileSize(_ size: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(size))
    }
}

// Component for displaying available models
struct AvailableModelsSection: View {
    let models: [Model]
    let onDownload: (Model) -> Void
    let onDelete: (Model) -> Void
    let onSelect: (Model) -> Void
    let onLoad: (Model) -> Void  // New parameter for loading a model
    let onShowInfo: (Model) -> Void  // New parameter for showing model info
    let onCancelDownload: (Model) -> Void
    let formatFileSize: (Int) -> String
    let modelService: ModelService
    @Binding var isDownloading: [String: Bool]
    @Binding var downloadProgress: [String: Double]
    @Binding var currentDownloadingFile: [String: String]
    let modelsWithIncompleteFiles: Set<String>
    let modelErrors: [String: String]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Check if any models are downloaded
            let anyModelsDownloaded = models.contains(where: { $0.isDownloaded })
            
            // Only show header if we have any models
            if !models.isEmpty {
                HStack {
                    Image(systemName: "square.stack.3d.down.right.fill")
                        .foregroundColor(.blue)
                        .font(.title3)
                    
                    if anyModelsDownloaded {
                        Text("Available Models")
                            .font(.headline)
                    } else {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Available Models")
                                .font(.headline)
                            Text("Select a model to download")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    Spacer()
                }
                .padding(.bottom, 4)
            }
            
            if models.isEmpty {
                Text("No models available")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
            } else {
                ForEach(models) { model in
                    ModelListItemHorizontal(
                        model: model,
                        isDownloaded: model.isDownloaded,
                        isDownloading: isDownloading[model.id] ?? false,
                        downloadProgress: downloadProgress[model.id] ?? 0.0,
                        currentFile: currentDownloadingFile[model.id] ?? "",
                        isSelected: modelService.isModelSelected(model),
                        onSelect: { onSelect(model) },
                        onLoad: { onLoad(model) },
                        onDelete: { onDelete(model) },
                        onDownload: { onDownload(model) },
                        onCancelDownload: { onCancelDownload(model) },
                        onShowInfo: { onShowInfo(model) },
                        hasIncompleteFiles: modelsWithIncompleteFiles.contains(model.id),
                        errorMessage: modelErrors[model.id]
                    )
                }
            }
        }
    }
}

// Component for a single model in the list
    struct CustomModelSection: View {
        @Binding var customModelURL: String
        @Binding var customModelName: String
        @Binding var isAddingCustomModel: Bool
        @Binding var isCustomModelSheetActive: Bool
        let onAddCustomModel: () -> Void
        
        var body: some View {
            VStack(alignment: .leading, spacing: 12) {
                Button(action: {
                    isAddingCustomModel = true
                    isCustomModelSheetActive = true
                }) {
                    HStack {
                        Image(systemName: "plus.circle.fill")
                            .foregroundColor(.blue)
                        Text("Add Custom Model")
                            .foregroundColor(.blue)
                            .font(.headline)
                        Spacer()
                        Image(systemName: "chevron.right")
                            .foregroundColor(.blue)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                }
                .buttonStyle(PlainButtonStyle())
            }
        }
    }

// Sheet for adding custom models
    struct CustomModelSheet: View {
        @Binding var customModelURL: String
        @Binding var customModelName: String
        @Binding var customModelDescription: String
        @Binding var isCustomModelSheetActive: Bool
        let onAddCustomModel: () -> Void
        @Binding var isAddingCustomModel: Bool
        @Environment(\.dismiss) private var dismiss
        
        var body: some View {
            NavigationView {
                Form {
                    Section(header: Text("Model Details")) {
                        TextField("Model URL or local path (required)", text: $customModelURL)
                            .autocapitalization(.none)
                            .disableAutocorrection(true)
                        
                        TextField("Model Name (optional)", text: $customModelName)
                        
                        TextField("Model Description (optional)", text: $customModelDescription)
                            .lineLimit(3)
                    }
                    
                    Section {
                        Button(action: {
                            isCustomModelSheetActive = false
                            onAddCustomModel()
                        }) {
                            Text("Add Model")
                            .frame(maxWidth: .infinity)
                            .foregroundColor(.white)
                        }
                        .frame(maxWidth: .infinity)
                        .listRowBackground(Color.blue)
                    }
                    
                    Section {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Custom models formatted for iOS/macOS app can be found here:")
                                .font(.footnote)
                            
                            Link("https://huggingface.co/collections/anemll/anemll-ios-67bdea29e45a1bf4b47d8623",
                                 destination: URL(string: "https://huggingface.co/collections/anemll/anemll-ios-67bdea29e45a1bf4b47d8623")!)
                                .font(.footnote)
                                .foregroundColor(.blue)
                        }
                        .padding(.vertical, 4)
                    }
                }
                .navigationTitle("Add Custom Model")
                .navigationBarItems(trailing: Button("Cancel") {
                isCustomModelSheetActive = false
                    isAddingCustomModel = false
                dismiss()
            })
            .onAppear {
                isCustomModelSheetActive = true
            }
            .onDisappear {
                isCustomModelSheetActive = false
            }
        }
    }
}

// Custom done button
extension ModelManagementView {
    var doneButton: some View {
        Button("Done") {
            // Dismiss keyboard before dismissing view
            dismissKeyboardAndCleanup()
            
            // Dismiss view
                    self.dismiss()
                }
    }
}

#Preview {
    ModelManagementView()
}

// MARK: - Extensions for Service Integration

// Extension to add URL formatting for models
extension Model {
    // Helper function to format source URLs for display
    func formatSourceURL(_ url: String) -> String {
        // Handle nil or empty URL
        guard !url.isEmpty else {
            return "No Source URL Available"
        }
        
        // If URL is a filesystem path, mask it
        if url.starts(with: "/") || url.starts(with: "~") || url.contains("SourceRelease") {
            // For local filesystem URLs, use a generic format
            return "${MODEL_ROOT}/models/custom/\(URL(fileURLWithPath: url).lastPathComponent)"
        } else if url.starts(with: "file://") {
            // Extract and mask file URLs
            let path = url.replacingOccurrences(of: "file://", with: "")
            return "${MODEL_ROOT}/models/custom/\(URL(fileURLWithPath: path).lastPathComponent)"
        }
        // For web URLs, return as is
        return url
    }
}

// Extension to provide compatibility with InferenceService
extension InferenceService {
    // Function to get the currently loaded model
    func getCurrentlyLoadedModel() -> Model? {
        guard isModelLoaded, let modelId = currentModelIdForUI else {
            return nil
        }
        
        // Look up the model in the ModelService by ID
        return ModelService.shared.getAvailableModels().first { $0.id == modelId }
    }
    
    // A public computed property to access the currentModelId safely 
    var currentModelIdForUI: String? {
        return self.isModelLoaded ? self.currentModel : nil
    }
}

// Extension to provide additional methods for ModelService
extension ModelService {
    // Check if a model has incomplete or corrupted files
    func hasIncompleteDownload(modelId: String) -> Bool {
        return !verifyModelFiles(modelId: modelId)
    }
    
    // Download only missing or corrupted files
    func downloadMissingFiles(model: Model, fileProgress: @escaping (String, Double) -> Void) {
        // This is a wrapper around downloadModel but with special flags
        downloadModel(modelId: model.id, fileProgress: fileProgress, completion: { _ in
            // Completion handled by caller
        })
    }
    
    // Force a complete redownload by deleting and redownloading
    func forceRedownload(model: Model, fileProgress: @escaping (String, Double) -> Void) {
        // First delete the model files
        let modelPath = getModelPath(for: model.id)
        if FileManager.default.fileExists(atPath: modelPath.path) {
            try? FileManager.default.removeItem(at: modelPath)
        }
        
        // Then download it again
        downloadModel(modelId: model.id, fileProgress: fileProgress, completion: { _ in
            // Completion handled by caller
        })
    }
}

// Add this new view after the other custom views
struct ModelRedownloadSheet: View {
    let model: Model
    let hasIncompleteDownload: Bool
    let actualSize: Int
    let sizePercentage: Float
    let onDownloadMissing: () -> Void
    let onDownloadAll: () -> Void
    let onForceRedownload: () -> Void
    @Binding var isPresented: Bool
    
    private func formatFileSize(_ size: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(size))
    }
    
    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Model Verification")) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(model.name)
                            .font(.headline)
                        
                        // Size verification info
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Size Verification")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            HStack {
                                Text("Expected size:")
                                    .foregroundColor(.secondary)
                                Spacer()
                                Text(formatFileSize(model.size))
                                    .foregroundColor(.secondary)
                            }
                            
                            HStack {
                                Text("Actual size on disk:")
                                    .foregroundColor(.secondary)
                                Spacer()
                                Text(formatFileSize(actualSize))
                                    .foregroundColor(sizePercentage < 98 ? .orange : .green)
                            }
                            
                            HStack {
                                Text("Completeness:")
                                    .foregroundColor(.secondary)
                                Spacer()
                                Text("\(String(format: "%.1f", sizePercentage))%")
                                    .foregroundColor(sizePercentage < 98 ? .orange : .green)
                            }
                        }
                        .padding(.vertical, 8)
                        
                        Divider()
                        
                        if hasIncompleteDownload {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Image(systemName: "exclamationmark.triangle.fill")
                                        .foregroundColor(.orange)
                                    Text("This model has incomplete or corrupted files")
                                        .foregroundColor(.orange)
                                        .font(.subheadline)
                                }
                                
                                Text("Missing weight files were detected. The model needs to be fixed before it can be used.")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        } else {
                            HStack {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                Text("This model is already downloaded")
                                    .foregroundColor(.green)
                                    .font(.subheadline)
                            }
                        }
                    }
                    .padding(.vertical, 8)
                }
                
                Section(header: Text("Download Options")) {
                    Button(action: {
                        isPresented = false
                        onDownloadAll()
                    }) {
                        HStack {
                            Image(systemName: "arrow.down.circle")
                                .foregroundColor(.blue)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Normal Redownload")
                                    .foregroundColor(.primary)
                                Text("Download all model files, preserving existing ones")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    
                    Button(action: {
                        isPresented = false
                        onForceRedownload()
                    }) {
                        HStack {
                            Image(systemName: "exclamationmark.arrow.down.circle")
                                .foregroundColor(.red)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Force Complete Redownload")
                                    .foregroundColor(.red)
                                Text("Delete all existing files and redownload from scratch")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
                
                Section {
                    Button(action: {
                        isPresented = false
                    }) {
                        Text("Cancel")
                            .foregroundColor(.blue)
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding(.vertical, 4)
                    }
                }
            }
            .listStyle(InsetGroupedListStyle())
            .navigationTitle(hasIncompleteDownload ? "Repair Model" : "Verify Model")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

// Model Information Sheet
struct ModelInfoSheet: View {
    let model: Model
    @Binding var isPresented: Bool
    @State private var additionalInfo: [String: String] = [:]
    @State private var isLoadingInfo: Bool = true
    @State private var showSizeVerification: Bool = false
    @State private var isVerifyingSizes: Bool = false
    @State private var sizeVerificationResults: String = ""
    @State private var sizeVerificationIsValid: Bool = true
    
    // Reference to shared ModelService
    let modelService = ModelService.shared
    
    // Helper function to format source URLs for display
    private func formatSourceURL(_ url: String) -> String {
        // If URL is a filesystem path, mask it
        if url.starts(with: "/") || url.starts(with: "~") || url.contains("SourceRelease") {
            // For local filesystem URLs, use a generic format
            return "${MODEL_ROOT}/models/custom/\(URL(fileURLWithPath: url).lastPathComponent)"
        } else if url.starts(with: "file://") {
            // Extract and mask file URLs
            let path = url.replacingOccurrences(of: "file://", with: "")
            return "${MODEL_ROOT}/models/custom/\(URL(fileURLWithPath: path).lastPathComponent)"
        }
        // For web URLs, return as is
        return url
    }
    
    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Basic Information")) {
                    infoRow(title: "Name", value: model.name)
                    infoRow(title: "ID", value: model.id)
                    infoRow(title: "Size", value: formatFileSize(model.size))
                    if !model.description.isEmpty {
                        infoRow(title: "Description", value: model.description)
                    }
                    infoRow(title: "Status", value: model.isDownloaded ? "Downloaded" : "Not Downloaded")
                }
                
                if isLoadingInfo {
                    Section(header: Text("Additional Details")) {
                        HStack {
                            Text("Loading information...")
                            Spacer()
                            ProgressView()
                        }
                    }
                } else if !additionalInfo.isEmpty {
                    Section(header: Text("Additional Details")) {
                        ForEach(additionalInfo.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                            infoRow(title: formatInfoKey(key), value: value)
                        }
                    }
                }
                
                if model.isDownloaded {
                    Section(header: Text("Verification")) {
                        Button(action: {
                            verifySizes()
                        }) {
                            HStack {
                                Text("Verify Model Size")
                                    .foregroundColor(.blue)
                                Spacer()
                                if isVerifyingSizes {
                                    ProgressView()
                                } else {
                                    Image(systemName: "chevron.right")
                                        .foregroundColor(.gray)
                                }
                            }
                        }
                        .disabled(isVerifyingSizes)
                    }
                    
                    if showSizeVerification {
                        Section(header: Text("Size Verification Results")) {
                            VStack(alignment: .leading, spacing: 10) {
                                HStack {
                                    Image(systemName: sizeVerificationIsValid ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                        .foregroundColor(sizeVerificationIsValid ? .green : .orange)
                                    Text(sizeVerificationIsValid ? "Size verification passed" : "Size verification found issues")
                                        .fontWeight(.medium)
                                        .foregroundColor(sizeVerificationIsValid ? .green : .orange)
                                }
                                
                                Text(sizeVerificationResults)
                                    .font(.system(.footnote, design: .monospaced))
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
                
                if let sourceURL = additionalInfo["source_url"], !sourceURL.isEmpty {
                    Section(header: Text("Source")) {
                        let displayURL = formatSourceURL(sourceURL)
                        
                        VStack(alignment: .leading) {
                            // Display the formatted URL text
                            Text(displayURL)
                                .font(.footnote)
                                .foregroundColor(.secondary)
                            
                            // Only make it clickable if it's a web URL
                            if !displayURL.contains("${MODEL_ROOT}") && sourceURL.hasPrefix("http") {
                                Link(destination: URL(string: sourceURL) ?? URL(string: "https://huggingface.co")!) {
                                    HStack {
                                        Text("Open Source URL")
                                            .foregroundColor(.blue)
                                        Spacer()
                                        Image(systemName: "arrow.up.right.square")
                                            .foregroundColor(.blue)
                                    }
                                }
                            } else {
                                Text("Local Custom Model")
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Model Information")
            .navigationBarItems(trailing: Button("Done") { isPresented = false })
            .onAppear {
                loadAdditionalInfo()
            }
        }
    }
    
    // Function to verify model sizes
    private func verifySizes() {
        isVerifyingSizes = true
        showSizeVerification = true
        
        // Run verification in background
        Task {
            let (isValid, _, sizeInfo) = modelService.verifyModelSizes(modelId: model.id)
            
            // Update UI on main thread
            await MainActor.run {
                sizeVerificationResults = sizeInfo
                sizeVerificationIsValid = isValid
                isVerifyingSizes = false
            }
        }
    }
    
    // Format file size in human-readable format
    private func formatFileSize(_ size: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(size))
    }
    
    // Format info key to be more user-friendly
    private func formatInfoKey(_ key: String) -> String {
        let formatted = key.replacingOccurrences(of: "_", with: " ")
            .capitalized
        return formatted
    }
    
    // Helper function for consistent row styling
    private func infoRow(title: String, value: String) -> some View {
        HStack {
            Text(title)
                .fontWeight(.medium)
            Spacer()
            Text(value)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }
    
    // Load additional information from models.json or YAML config
    private func loadAdditionalInfo() {
        Task {
            isLoadingInfo = true
            var info: [String: String] = [:]
            
            // Different approach based on model type
            if isPredefinedModel(model.id) {
                // For predefined models, load hardcoded info
                info = getPredefinedModelInfo(model.id)
            } else {
                // For custom models, try multiple sources to get the URL
                
                // First check if there's a source_url in model downloadURL property
                if !model.downloadURL.isEmpty && model.downloadURL.hasPrefix("http") {
                    info["source_url"] = model.downloadURL
                }
                
                // If that failed, try to get info from meta.yaml
                if info["source_url"] == nil || info["source_url"]?.isEmpty == true {
                    if let sourceURL = modelService.getSourceURLFromMetaYaml(for: model.id) {
                        info["source_url"] = sourceURL
                    }
                }
                
                // If still no URL and it's a custom model, use a placeholder
                if (info["source_url"] == nil || info["source_url"]?.isEmpty == true) && model.id.contains("custom") {
                    info["source_url"] = "file://\(modelService.getModelPath(for: model.id).path)"
                }
                
                // Try to get additional info from model directory
                if model.isDownloaded {
                    let modelPath = modelService.getModelPath(for: model.id)
                    info = await getModelInfoFromMetadata(modelPath: modelPath, info: info)
                }
            }
            
            // Update on main thread
            await MainActor.run {
                self.additionalInfo = info
                self.isLoadingInfo = false
            }
        }
    }
    
    // Check if a model is a predefined model
    private func isPredefinedModel(_ modelId: String) -> Bool {
        return modelId == "llama-3.2-1b" || 
               modelId.contains("llama_3_2_1b") || 
               modelId.contains("llama-3.2")
    }
    
    // Get predefined info for built-in models
    private func getPredefinedModelInfo(_ modelId: String) -> [String: String] {
        // Hardcoded info for the default Llama 3.2 1B model
        if modelId == "llama-3.2-1b" || 
           modelId.contains("llama_3_2_1b") || 
           modelId.contains("llama-3.2") {
            return [
                "architecture": "Llama 3.2",
                "parameters": "1.1 billion",
                "context_length": "1024 tokens",
                "quantization": "4-bit",
                "source_url": "https://huggingface.co/anemll/anemll-llama-3.2-1B-iOSv2.0",
                "license": "Meta Llama 3 Community License",
                "optimized_for": "iOS and macOS devices"
            ]
        }
        
        return [:]
    }
    
    // Get info from meta.yaml or other metadata files
    private func getModelInfoFromMetadata(modelPath: URL, info: [String: String]) async -> [String: String] {
        var updatedInfo = info
        
        // Check for meta.yaml
        let metaYamlPath = modelPath.appendingPathComponent("meta.yaml")
        
        do {
            // Read file content in background thread - don't use try with Task.detached, only with its result
            let contentOptional = await Task.detached(priority: .background) {
                do {
                    return try String(contentsOf: metaYamlPath, encoding: .utf8)
                } catch {
                    print("Error reading meta.yaml: \(error)")
                    return "" // Return empty string instead of nil
                }
            }.value
            
            // Only proceed if we have content
            if !contentOptional.isEmpty {
                // Parse the YAML content for key information
                let lines = contentOptional.components(separatedBy: CharacterSet.newlines)
                for line in lines {
                    if line.contains(":") {
                        let components = line.components(separatedBy: ":")
                        if components.count >= 2 {
                            let key = components[0].trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                            let value = components[1].trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                            if !key.isEmpty && !value.isEmpty {
                                updatedInfo[key] = value
                            }
                        }
                    }
                }
            }
            
            // Add a potential throw to make catch block reachable
            if contentOptional.isEmpty {
                throw NSError(domain: "ModelManagementView", code: 1, userInfo: [NSLocalizedDescriptionKey: "No content in meta.yaml"])
            }
        } catch {
            print("Error loading model metadata: \(error)")
        }
        
        return updatedInfo
    }
}

// Helper component for consistent button styling
struct ActionButton: View {
    let title: String
    let backgroundColor: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline)
                .fontWeight(.medium)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .frame(minWidth: 70)
                .background(backgroundColor)
                .foregroundColor(.white)
                .cornerRadius(8)
        }
    }
}

