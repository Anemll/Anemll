// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// ChatView.swift

import SwiftUI
import AVFoundation
import AudioToolbox
import Combine // For Combine functionality

struct BottomYPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat? = nil
    static func reduce(value: inout CGFloat?, nextValue: () -> CGFloat?) {
        value = nextValue()
    }
}

struct ChatView: View {
    @StateObject var chat: Chat
    @State private var inputText: String = ""
    @State private var isTyping: Bool = false
    @State private var showModelManagement = false
    @State private var showSettings = false
    @State private var showModelLoadingError = false
    @State private var errorMessage = ""
    @State private var keyboardHeight: CGFloat = 0
    @State private var inferenceTask: Task<Void, Never>? = nil
    @State private var contentHeight: CGFloat = 0
    @State private var scrollViewHeight: CGFloat = 0
    @State private var isAtBottom: Bool = true
    @State private var scrollingTimer: Timer? = nil
    @State private var scrollProxy: ScrollViewProxy? = nil
    @State private var showCopiedFeedback: Bool = false
    @State private var forceScrollTrigger = false // Add a trigger for forcing scrolls
    @State private var showReloadConfirmation = false
    @State private var currentModelId: String? = nil
    @State private var isCancellingModel = false
    @State private var allowLongGeneration = true
    @State private var currentWindowShifts: Int = 0
    @State private var isAdvancedMode = false
    @State private var showingClearAlert = false
    @State private var showSettingsPopover = false // New state for custom popover
    @State private var settingsAnchor: CGPoint = .zero // Anchor point for popover
    @State private var useRepetitionDetector = InferenceService.shared.isRepetitionDetectorEnabled
    @ObservedObject private var modelService = ModelService.shared
    @ObservedObject private var inferenceService = InferenceService.shared
    @ObservedObject private var chatService = ChatService.shared
    @FocusState private var isInputFocused: Bool

    var isModelLoading: Bool {
        guard let selectedModel = modelService.getSelectedModel() else { return false }
        return selectedModel.isDownloaded && (!inferenceService.isModelLoaded || inferenceService.currentModel != selectedModel.id)
    }

    // Add an initializer that logs when the ChatView is created
    init(chat: Chat) {
        print("DEBUG: Initializing ChatView for chat ID: \(chat.id), messages count: \(chat.messages.count)")
        _chat = StateObject(wrappedValue: chat)
    }

    var body: some View {
        mainContent
            .navigationTitle("")
            .toolbar { toolbarContent }
            .sheet(isPresented: $showSettings) { SettingsView() }
            .sheet(isPresented: $showModelManagement, onDismiss: {
                print("DEBUG: ModelManagementView dismissed from ChatView")
                
                // Inform the coordinator that the view has been dismissed
                ModelManagementCoordinator.didDismiss()
                
                // Check if we still need a model after dismissal
                let needModel = modelService.getSelectedModel() == nil || 
                              (modelService.getSelectedModel() != nil && 
                               !modelService.verifyModelFiles(modelId: modelService.getSelectedModel()!.id))
                
                if needModel {
                    print("DEBUG: Model still required after dismiss attempt from ChatView")
                    // Add delay before attempting to show again to avoid presentation conflicts
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        Task { @MainActor in
                            if ModelManagementCoordinator.canPresent() {
                                print("DEBUG: Re-showing ModelManagementView after failed dismiss")
                                showModelManagement = true
                            }
                        }
                    }
                } else {
                    // Keep the view dismissed
                    print("DEBUG: Model verification successful, ChatView ModelManagementView dismissed")
                }
            }) { 
                // Create a new instance each time with a unique ID to avoid state persistence issues
                ModelManagementView()
                    .id(UUID()) // Force a fresh instance each time
                    .interactiveDismissDisabled(modelService.getSelectedModel() == nil || 
                                              (modelService.getSelectedModel() != nil && 
                                               !modelService.verifyModelFiles(modelId: modelService.getSelectedModel()!.id))) // Only allow dismiss if there's a valid model
                    .onAppear {
                        print("DEBUG: ModelManagementView appeared from ChatView")
                        // Ensure keyboard is dismissed when sheet appears
                        dismissKeyboard()
                    }
                    .onDisappear {
                        print("DEBUG: ModelManagementView disappeared from ChatView")
                    }
            }
            .alert("Model Loading Error", isPresented: $showModelLoadingError) {
                Button("OK", role: .cancel) { }
            } message: { Text(errorMessage) }
            .onAppear { 
                // Ensure we clear any stored presentation state
                ModelManagementCoordinator.didDismiss()
                
                // Use Task to safely handle setup
                Task { @MainActor in
                    print("DEBUG: ChatView appeared for chat ID: \(chat.id), messages count: \(chat.messages.count)")
                    setupKeyboardObservers() 
                    
                    // Ensure model indicator is immediately visible with fallback state
                    print("DEBUG: Ensuring model indicator is visible with fallback state")
                    print("DEBUG: Current model status - selected model: \(modelService.getSelectedModel()?.name ?? "none")")
                    
                    // Force UI refresh for consistent layout immediately
                    forceScrollTrigger.toggle()
                    
                    // Check model availability on the next run loop after rendering
                    try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
                    
                    print("DEBUG: Checking model availability on appear")
                    checkModelAvailability()
                    
                    // Ensure input field has focus after UI is fully initialized
                    try? await Task.sleep(nanoseconds: 300_000_000) // additional 0.3 seconds
                    
                    // Ensure we have focus in the input field
                    isInputFocused = true
                }
            }
            .onDisappear { 
                // Use Task to safely handle cleanup
                Task { @MainActor in
                    print("DEBUG: ChatView disappeared for chat ID: \(chat.id)")
                    removeKeyboardObservers() 
                }
            }
            .overlay(copiedFeedbackOverlay)
            .alert("Clear Conversation", isPresented: $showingClearAlert) {
                Button("Cancel", role: .cancel) { }
                Button("Clear", role: .destructive) {
                    // Clear the current conversation
                    chat.messages = []
                    chatService.saveChat(chat)
                    print("DEBUG: Conversation cleared")
                }
            } message: {
                Text("Are you sure you want to clear all messages in this conversation? This action cannot be undone.")
            }
            .onChange(of: chat.messages) { oldMessages, newMessages in
                print("DEBUG: Chat messages changed for chat ID: \(chat.id), old count: \(oldMessages.count), new count: \(newMessages.count)")
            }
    }
    
    private var mainContent: some View {
        VStack(spacing: 0) {
            // Ensure the model indicator is always shown regardless of model state
            ZStack(alignment: .leading) {
                // Fallback view that will always appear
                HStack {
                    Image(systemName: "cpu").foregroundColor(.gray)
                    Text("Model: N/A").font(.caption).foregroundColor(.secondary)
                    Spacer()
                }
                .padding(.horizontal).padding(.vertical, 4)
                .frame(maxWidth: .infinity)
                .background(Color(.systemGray6))
                
                // Actual model indicator that will overlay the fallback when available
                if modelService.getSelectedModel() != nil {
                    modelIndicatorView
                        .frame(maxWidth: .infinity)
                }
            }
            .frame(maxWidth: .infinity)
            .fixedSize(horizontal: false, vertical: true)
            
            // Messages view in a separate container
            MessagesContainerView(
                chat: chat,
                isTyping: $isTyping,
                showCopiedFeedback: $showCopiedFeedback,
                isAtBottom: $isAtBottom,
                scrollingTimer: $scrollingTimer,
                scrollProxy: $scrollProxy,
                contentHeight: $contentHeight,
                scrollViewHeight: $scrollViewHeight,
                forceScrollTrigger: $forceScrollTrigger // Pass the force scroll trigger
            )
            .padding()
            .onAppear {
                print("DEBUG: MessagesContainerView appeared for chat ID: \(chat.id)")
            }
        }
        .safeAreaInset(edge: .bottom) {
            inputArea
        }
    }
    
    private var modelIndicatorView: some View {
        ModelIndicatorView(
            modelService: modelService,
            inferenceService: inferenceService,
            showModelManagement: $showModelManagement,
            isCancellingModel: isCancellingModel,
            onReloadModel: {
                // Try to get model ID from currentModel first
                var modelToReload: Model? = nil
                
                // Store the current model ID for reload if available
                if let modelId = inferenceService.currentModel, 
                   let model = modelService.getModel(for: modelId) {
                    modelToReload = model
                    currentModelId = modelId
                } 
                // If no current model, try to get the selected model
                else if let selectedModel = modelService.getSelectedModel() {
                    modelToReload = selectedModel
                    currentModelId = selectedModel.id
                }
                
                // Now proceed with reloading if we have a model
                if let model = modelToReload {
                    // If model is loaded or loading, show confirmation
                    if inferenceService.isModelLoaded || inferenceService.isLoadingModel {
                        showReloadConfirmation = true
                    } else {
                        // If model is selected but not loaded, reload directly
                        ModelService.shared.loadModel(model)
                    }
                }
                // Never show model management view from this button
            }
        )
    }
    
    private var inputArea: some View {
        HStack {
            messageTextField
            
            actionButton
        }
        .padding()
        .background(Color(.systemBackground))
    }
    
    private var messageTextField: some View {
        TextField("Type a message...", text: $inputText, onCommit: {
            if !inputText.isEmpty && !isTyping && !isModelLoading { Task { await sendMessage() } }
        })
        .padding(10)
        .background(Color.gray.opacity(0.3))
        .cornerRadius(20)
        .focused($isInputFocused)
        .disabled(isTyping)
        .submitLabel(.send)
    }
    
    private var actionButton: some View {
        Group {
            Button(action: handleSendTap) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 30))
                    .foregroundColor(inputText.isEmpty || isTyping ? .gray : .blue)
            }
            .disabled(inputText.isEmpty || isTyping)
            
            if isTyping {
                Button(action: stopInference) {
                    Image(systemName: "stop.fill")
                        .foregroundColor(.red)
                        .font(.system(size: 30))
                }
            }
        }
        .alert("Reload Model", isPresented: $showReloadConfirmation) {
            Button("Cancel", role: .cancel) { }
            Button("Reload", role: .destructive) {
                reloadCurrentModel()
            }
        } message: {
            Text("Do you want to reload the current model? This will interrupt any ongoing tasks.")
        }
    }
    
    private var toolbarContent: some ToolbarContent {
        ToolbarItemGroup(placement: .navigationBarTrailing) {
            // Thinking mode toggle button
            Button(action: {
                let currentThinking = inferenceService.isThinkingModeEnabled()
                inferenceService.setThinkingMode(!currentThinking)
                
                // Add a system message to inform the user about the toggle
                let statusMessage = Message(
                    text: "Thinking mode \(!currentThinking ? "enabled" : "disabled")",
                    isUser: false,
                    isSystemMessage: true
                )
                chat.addMessage(statusMessage)
                chatService.saveChat(chat)
                
                // Scroll to bottom to show the status message
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    Task { @MainActor in
                        self.scrollToBottom()
                    }
                }
            }) {
                Image(systemName: "brain")
                    .foregroundColor(inferenceService.isThinkingModeEnabled() ? .blue : .gray)
                    .fontWeight(inferenceService.isThinkingModeEnabled() ? .bold : .regular)
            }
            .disabled(isTyping)
            
            Button {
                // Use a dedicated method to handle showing the model management view
                showModelManagementView()
            } label: {
                Image(systemName: "cube")
                    .foregroundColor(.blue)
            }
            
            // Replace Menu with a Button that shows custom popover
            Button {
                // Show settings popover
                showSettingsPopover = true
                print("DEBUG: Settings button tapped, showing popover")
            } label: {
                Label("Settings", systemImage: "gear")
            }
            .background(
                GeometryReader { geo in
                    Color.clear
                        .onAppear {
                            // Get the position for the popover
                            let frame = geo.frame(in: .global)
                            settingsAnchor = CGPoint(x: frame.midX, y: frame.midY)
                        }
                }
            )
            .popover(isPresented: $showSettingsPopover, arrowEdge: .top) {
                // Custom popover content
                VStack(spacing: 16) {
                    Text("Settings")
                        .font(.headline)
                        .padding(.top)
                    
                    Divider()
                    
                    VStack(alignment: .leading) {
                        Toggle("Show advanced commands", isOn: $isAdvancedMode)
                            .toggleStyle(SwitchToggleStyle(tint: .blue))
                            .padding(.horizontal)
                            .onAppear {
                                print("DEBUG: Advanced Mode toggle value: \(isAdvancedMode)")
                            }
                            .onChange(of: isAdvancedMode) { oldValue, newValue in
                                print("DEBUG: Advanced Mode toggle changed to: \(newValue)")
                            }
                        
                        Toggle("Long Generation", isOn: $allowLongGeneration)
                            .toggleStyle(SwitchToggleStyle(tint: .green))
                            .padding(.horizontal)
                            .onAppear {
                                print("DEBUG: Long Generation toggle value: \(allowLongGeneration)")
                            }
                            .onChange(of: allowLongGeneration) { oldValue, newValue in
                                print("DEBUG: Long Generation toggle changed to: \(newValue)")
                            }
                        
                        Text("Enables responses up to 4x longer with extended token limits")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.horizontal)
                            
                        Toggle("Repetition Detection", isOn: $useRepetitionDetector)
                            .toggleStyle(SwitchToggleStyle(tint: .purple))
                            .padding(.horizontal)
                            .onAppear {
                                print("DEBUG: Repetition Detection toggle value: \(useRepetitionDetector)")
                            }
                            .onChange(of: useRepetitionDetector) { oldValue, newValue in
                                print("DEBUG: Repetition Detection toggle changed to: \(newValue)")
                                InferenceService.shared.configureRepetitionDetector(enabled: newValue)
                            }
                        
                        Text("Automatically stops generation if text becomes repetitive")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.horizontal)
                    }
                    
                    if isAdvancedMode {
                        Divider()
                        
                        // InferenceManager Debug Controls
                        VStack(spacing: 8) {
                            Text("Debug Controls")
                                .font(.headline)
                                .padding(.vertical, 4)
                            
                            Button("Reset Backings") {
                                if let inferenceManager = inferenceService.debugInferenceManager {
                                    Task {
                                        do {
                                            try await inferenceManager.initializeBackings()
                                        } catch {
                                            print("Error resetting backings: \(error)")
                                        }
                                    }
                                }
                            }
                            .disabled(inferenceService.debugInferenceManager == nil)
                            
                            Button("Reset Causal Mask") {
                                if let inferenceManager = inferenceService.debugInferenceManager {
                                    inferenceManager.initFullCausalMask()
                                }
                            }
                            .disabled(inferenceService.debugInferenceManager == nil)
                            
                            Button("Reset KV-Cache") {
                                if let inferenceManager = inferenceService.debugInferenceManager {
                                    inferenceManager.initState()
                                }
                            }
                            .disabled(inferenceService.debugInferenceManager == nil)
                            
                            Button("Toggle Debug Level") {
                                if let inferenceManager = inferenceService.debugInferenceManager {
                                    inferenceManager.ToggeDebugLevel()
                                }
                            }
                            .disabled(inferenceService.debugInferenceManager == nil)
                        }
                        
                        Divider()
                        
                        Button("Clear Conversation") {
                            showSettingsPopover = false
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                showingClearAlert = true
                            }
                        }
                        .foregroundColor(.red)
                        .padding(.bottom)
                    }
                }
                .frame(width: 250)
                .padding(.vertical, 10)
            }
        }
    }
    
    private var copiedFeedbackOverlay: some View {
        Group {
            if showCopiedFeedback {
                Text("Copied!")
                    .padding()
                    .background(Color.black.opacity(0.7))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .transition(.opacity)
            }
        }
    }
    
    // MARK: - Actions
    
    private func handleSendTap() {
        if isModelLoading {
            playErrorSound()
            showModelLoadingToast()
        } else if !inputText.isEmpty {
            Task { await sendMessage() }
        }
    }
    
    private func removeKeyboardObservers() {
        NotificationCenter.default.removeObserver(self, name: UIResponder.keyboardWillShowNotification, object: nil)
        NotificationCenter.default.removeObserver(self, name: UIResponder.keyboardWillHideNotification, object: nil)
    }
    
    // MARK: - Private Methods

    private func setupKeyboardObservers() {
        NotificationCenter.default.addObserver(
            forName: UIResponder.keyboardWillShowNotification,
            object: nil,
            queue: .main) { notification in
                if let keyboardFrame = notification.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? CGRect {
                    keyboardHeight = keyboardFrame.height
                }
            }
        
        NotificationCenter.default.addObserver(
            forName: UIResponder.keyboardWillHideNotification,
            object: nil,
            queue: .main) { _ in keyboardHeight = 0 }
    }

    private func dismissKeyboard() {
        isInputFocused = false
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }

    private func playErrorSound() {
        AudioServicesPlaySystemSound(1521)
    }

    private func showModelLoadingToast() {
        // Placeholder for toast implementation
        print("Model is loading, please wait")
    }

    private func scrollToBottom() {
        if let proxy = scrollProxy {
            // Wrap animation in a Task to avoid view update cycles
            Task { @MainActor in
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
        } else {
            // If we don't have a proxy yet, use the trigger to force a scroll
            Task { @MainActor in
                forceScrollTrigger.toggle()
            }
        }
    }

    private func sendMessage() async {
        // Don't send empty messages
        if inputText.isEmpty { return }
        
        // Check if model is loaded and ready
        if isModelLoading {
            showModelLoadingError = true
            errorMessage = "Please wait for the model to finish loading."
            return
        }
        
        // Check if inference is busy
        if let inferenceManager = inferenceService.debugInferenceManager, inferenceManager.isBusy() {
            showModelLoadingError = true
            errorMessage = "Please wait for the current generation to complete."
            return
        }
        
        if !inferenceService.isModelLoaded {
            if let selectedModel = modelService.getSelectedModel() {
                if selectedModel.isDownloaded {
                    // Check if model has incomplete files
                    let hasIncompleteFiles = modelService.hasIncompleteDownload(modelId: selectedModel.id)
                    if hasIncompleteFiles {
                        showModelLoadingError = true
                        errorMessage = "Model has missing files. Please verify and repair the model in Model Management."
                        
                        // Show model management after a short delay
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                            Task { @MainActor in
                                if ModelManagementCoordinator.canPresent() {
                                    self.showModelManagement = true
                                }
                            }
                        }
                        
                        return
                    }
                    
                    showModelLoadingError = true
                    errorMessage = "Model needs to be loaded. Please try again."
                    
                    // Attempt to load the model
                    Task {
                        do {
                            try await modelService.loadModelForInference(selectedModel)
                        } catch {
                            print("CV.1 Error loading model: \(error)")
                        }
                    }
                    
                    return
                } else {
                    showModelLoadingError = true
                    errorMessage = "Selected model is not downloaded. Please download it in Model Management."
                    return
                }
            } else {
                showModelLoadingError = true
                errorMessage = "No model selected. Please select a model in Settings."
                return
            }
        }
        
        // If we're already generating, don't start another
        if isTyping || inferenceTask != nil {
            return
        }
        
        // Add the user message to the chat
        print("Sending message: \(inputText)")
        
        let userMessage = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        inputText = ""
        
        // Check if the input is the "/t" command to toggle thinking mode
        if userMessage == "/t" {
            let currentThinking = inferenceService.isThinkingModeEnabled()
            inferenceService.setThinkingMode(!currentThinking)
            
            // Add a system message to inform the user about the toggle
            let statusMessage = Message(
                text: "Thinking mode \(!currentThinking ? "enabled" : "disabled")",
                isUser: false,
                isSystemMessage: true
            )
            chat.addMessage(statusMessage)
            chatService.saveChat(chat)
            
            // Scroll to bottom to show the status message
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                Task { @MainActor in
                    self.scrollToBottom()
                }
            }
            return
        }
        
        // Start a new task for handling the response
        inferenceTask = Task {
            // Check if we need to load the model
            if let selectedModel = modelService.getSelectedModel(), 
               inferenceService.currentModel != selectedModel.id {
                // Check for incomplete files before attempting to load
                let hasIncompleteFiles = modelService.hasIncompleteDownload(modelId: selectedModel.id)
                if hasIncompleteFiles {
                    await MainActor.run {
                        showModelLoadingError = true
                        errorMessage = "Model has missing files. Please verify and repair the model in Model Management."
                        
                        // Show model management after a short delay
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                            Task { @MainActor in
                                if ModelManagementCoordinator.canPresent() {
                                    self.showModelManagement = true
                                }
                            }
                        }
                    }
                    return
                }
                
                do {
                    try await modelService.loadModelForInference(selectedModel)
                } catch {
                    await MainActor.run {
                        showModelLoadingError = true
                        errorMessage = "Failed to load model: \(error.localizedDescription)"
                        return
                    }
                }
            }
            
            // Set the typing indicator
            await MainActor.run {
                isTyping = true
            }
            
            // Add user message to chat
            await MainActor.run {
                let userMessageObj = Message(text: userMessage, isUser: true)
                chat.addMessage(userMessageObj)
                
                let aiMessageObj = Message(text: "", isUser: false)
                chat.addMessage(aiMessageObj)
                
                // Cancel any existing timer
                scrollingTimer?.invalidate()
                
                // Use a more reliable timer with shorter interval for more responsive scrolling
                // Create the timer on the main thread but dispatch the state changes safely
                scrollingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
                    // Use Task to ensure state modifications happen outside view update cycle
                    Task { @MainActor in
                        self.scrollToBottom()
                    }
                }
                
                // Ensure immediate scroll after adding messages
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    Task { @MainActor in
                        self.scrollToBottom()
                    }
                }
                
                // Save chat state
                chatService.saveChat(chat)
            }
            
            // Ensure we're not cancelled
            if Task.isCancelled { return }
            
            if chat.lastAIMessageIndex() != nil {
                // Handle response generation
                do {
                    do {
                        // Pass empty system prompt for now, but this could be configured in settings later
                        let stream = try await chatService.generateResponseWithSystemPrompt(
                            for: chat, 
                            systemPrompt: "", 
                            userMessage: userMessage,
                            allowLongGeneration: allowLongGeneration
                        )
                        var fullText = ""
                        var lastUpdateTime = Date()
                        var currentTokensPerSecond = 0.0
                        var currentWindowShifts: Int = 0
                        
                        for try await response in stream {
                            // Check if task was cancelled
                            if Task.isCancelled { break }
                            
                            // Update with the full text each time to ensure consistent rendering
                            fullText = response.text
                            // Store token count but we don't need to read it later
                            _ = response.tokenCount
                            currentTokensPerSecond = response.tokensPerSecond
                            
                            // Track window shifts for UI indicator
                            if let shifts = response.windowShifts, shifts > currentWindowShifts {
                                currentWindowShifts = shifts
                                // Optionally display a notification about window shifting
                                if currentWindowShifts == 1 {
                                    // First shift - show a more detailed message
                                    print("Window shifted to handle long generation")
                                }
                            }
                            
                            // Only update the UI at most every 100ms for better fluidity
                            let now = Date()
                            if now.timeIntervalSince(lastUpdateTime) > 0.1 || response.isComplete {
                                lastUpdateTime = now
                                
                                // Append to the response text
                                if let index = chat.lastAIMessageIndex() {
                                    chat.messages[index].text = fullText
                                    // Also update token generation speed
                                    if currentTokensPerSecond > 0 {
                                        chat.messages[index].tokensPerSecond = currentTokensPerSecond
                                    }
                                    chat.messages[index].windowShifts = currentWindowShifts
                                }
                            }
                        }
                        
                        // Make sure to update with the final result
                        if let index = chat.lastAIMessageIndex() {
                            await MainActor.run {
                                let _ = chat.updateMessage(at: index, with: fullText)
                                
                                // Store the tokens per second only at the end of generation
                                chat.messages[index].tokensPerSecond = currentTokensPerSecond
                                
                                // Final scroll to bottom after message is complete
                                self.scrollToBottom()
                            }
                        }
                    } catch is CancellationError {
                        print("Inference cancelled")
                    } catch {
                        print("Inference error: \(error.localizedDescription)")
                        if let index = chat.lastAIMessageIndex() {
                            // Handle error on the main thread
                            await MainActor.run {
                                let _ = chat.updateMessage(at: index, with: "Error: \(error.localizedDescription)")
                                
                                // Final scroll even on error
                                self.scrollToBottom()
                            }
                        }
                    }
                }
            }
            
            // Wait to ensure the message update has been applied
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                // After ensuring the message is updated, mark typing as complete
                inferenceTask = nil
                isTyping = false
                
                // Set focus back to input field so user can type next message
                isInputFocused = true
                
                // Cancel any scroll timer
                scrollingTimer?.invalidate()
                scrollingTimer = nil
                
                // Wait a moment before saving to ensure UI updates are complete
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    chatService.saveChat(chat)
                    // One final scroll to ensure we're at the bottom
                    self.scrollToBottom()
                }
            }
        }
    }

    private func stopInference() {
        print("🛑 Stopping inference from ChatView")
        
        // Cancel the task that updates the UI
        inferenceTask?.cancel()
        inferenceTask = nil
        
        // Cancel only the token generation without unloading the model
        Task {
            // Use the dedicated method for cancelling token generation
            InferenceService.shared.cancelTokenGeneration()
            
            // Update UI state on the main thread
            await MainActor.run {
                isTyping = false
                
                // Cancel any scroll timer
                scrollingTimer?.invalidate()
                scrollingTimer = nil
                
                // Save the chat state with whatever we've generated so far
                chatService.saveChat(chat)
                
                print("🛑 Inference successfully stopped")
                
                // Return focus to input field after stopping inference
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                    self.isInputFocused = true
                }
            }
        }
    }
    
    private func reloadCurrentModel() {
        // Get the current model
        guard let modelId = currentModelId, 
              let model = ModelService.shared.getModel(for: modelId) else {
            // If no model is currently loaded, show model management
            showModelManagement = true
            return
        }
        
        print("Reloading model: \(model.name)")
        
        // First cancel any current loading
        if inferenceService.isLoadingModel {
            // Set cancellation in progress flag
            isCancellingModel = true
            
            // Start the cancellation
            inferenceService.cancelModelLoading(reason: .startingNewModel)
            
            // Start a timer to periodically check if cancellation is complete
            let startTime = Date()
            let timeout: TimeInterval = 15.0 // 15 second timeout
            
            // Store references to captured state before entering closure
            let modelRef = model
            
            // Create a repeating timer to check cancellation status
            let checkTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { timer in
                // Use Task to switch to the main actor context for reading the actor-isolated property
                Task { @MainActor in
                    let isStillLoading = self.inferenceService.isLoadingModel
                    let elapsed = Date().timeIntervalSince(startTime)
                    
                    // If loading is no longer in progress or we timed out
                    if !isStillLoading || elapsed > timeout {
                        // Stop the timer
                        timer.invalidate()
                        
                        // Reset cancellation flag since we're already on the main actor
                        self.isCancellingModel = false
                        
                        // If we've timed out but model is still loading, log a warning
                        if isStillLoading {
                            print("⚠️ WARNING: Timed out waiting for model cancellation after \(elapsed) seconds. Proceeding anyway.")
                        } else {
                            print("✅ Model cancellation completed after \(elapsed) seconds")
                        }
                        
                        // Start loading the model regardless of whether we timed out
                        // We're already on the main actor
                        ModelService.shared.loadModel(modelRef)
                    } else {
                        print("⏳ Still waiting for model cancellation... (\(Int(elapsed)) seconds elapsed)")
                    }
                }
            }
            
            // Safety mechanism: ensure the timer gets invalidated if something goes wrong
            DispatchQueue.main.asyncAfter(deadline: .now() + timeout + 1) {
                if checkTimer.isValid {
                    print("⚠️ Safety timeout reached for cancellation timer. Invalidating.")
                    checkTimer.invalidate()
                    
                    // Reset cancellation flag on main thread
                    Task { @MainActor in
                        self.isCancellingModel = false
                    }
                }
            }
        } else {
            // If nothing is loading, just load the model immediately
            // Call loadModel on the main thread since it's a @MainActor method
            Task { @MainActor in
                ModelService.shared.loadModel(model)
            }
        }
    }

    // MARK: - Helper Methods
    
    // Static coordinator for model management view presentation across the app
    private struct ModelManagementCoordinator {
        static var isBeingPresented = false
        static var lastPresentationTime: Date? = nil
        
        static func canPresent() -> Bool {
            // If already being presented, don't present again
            if isBeingPresented { return false }
            
            // If was presented recently, don't present again too quickly
            if let lastTime = lastPresentationTime, 
               Date().timeIntervalSince(lastTime) < 1.0 {
                return false
            }
            
            // Otherwise, allow presentation
            isBeingPresented = true
            lastPresentationTime = Date()
            return true
        }
        
        static func didDismiss() {
            isBeingPresented = false
        }
    }
    
    /// Checks if a valid model is available and shows the model management view if needed
    private func checkModelAvailability() {
        print("DEBUG: Checking model availability in ChatView")
        
        // Check if a model is selected
        guard let selectedModel = modelService.getSelectedModel() else {
            print("DEBUG: No model selected, showing model management view")
            showModelManagementView()
            return
        }
        
        // Check if the selected model is properly downloaded and verified
        let isModelValid = modelService.verifyModelFiles(modelId: selectedModel.id)
        if !isModelValid {
            print("DEBUG: Selected model verification failed, showing model management view")
            showModelManagementView()
            return
        }
        
        print("DEBUG: Model check passed for \(selectedModel.name)")
    }
    
    private func showModelManagementView() {
        // Dismiss keyboard first
        dismissKeyboard()
        
        // Use Task to safely dispatch UI updates
        Task { @MainActor in
            if !showModelManagement {
                if ModelManagementCoordinator.canPresent() {
                    print("DEBUG: Showing ModelManagementView from ChatView")
                    showModelManagement = true
                } else {
                    print("DEBUG: ModelManagementView presentation prevented by coordinator in ChatView")
                }
            } else {
                print("DEBUG: ModelManagementView already showing in ChatView, ignoring tap")
            }
        }
    }
}

// MARK: - Supporting Views

struct ModelIndicatorView: View {
    @ObservedObject var modelService: ModelService
    @ObservedObject var inferenceService: InferenceService
    @Binding var showModelManagement: Bool
    var isCancellingModel: Bool
    var onReloadModel: () -> Void
    @State private var showErrorDetails = false
    
    var isModelLoading: Bool {
        guard let selectedModel = modelService.getSelectedModel() else { return false }
        return selectedModel.isDownloaded && (!inferenceService.isModelLoaded || inferenceService.currentModel != selectedModel.id) && !inferenceService.hasLoadingError
    }
    
    // Helper method to safely show model management view
    private func showModelManagementView() {
        // Dismiss any active keyboard first to prevent it showing behind the sheet
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        
        // Use Task to safely dispatch UI updates
        Task { @MainActor in
            if !showModelManagement {
                // Simply set the binding to true to show the sheet
                print("DEBUG: ModelIndicatorView showing model management")
                showModelManagement = true
            }
        }
    }
    
    var body: some View {
        // Always provide content regardless of model state
        HStack {
            // When model is available, show full UI
            if let selectedModel = modelService.getSelectedModel() {
                // Make the CPU icon and model name clickable to open Model Management
                Button(action: { showModelManagementView() }) {
                    HStack {
                        Image(systemName: "cpu").foregroundColor(.blue)
                        Text("Model: \(selectedModel.name)").font(.caption).foregroundColor(.secondary)
                    }
                }
                .buttonStyle(PlainButtonStyle()) // Keep the default appearance
                
                Spacer()
                
                if inferenceService.isModelLoaded && inferenceService.currentModel == selectedModel.id {
                    Image(systemName: "checkmark.circle.fill").foregroundColor(.green).font(.caption)
                } else if isCancellingModel {
                    // Show cancellation state
                    HStack(spacing: 4) {
                        ProgressView()
                            .scaleEffect(0.5)
                        
                        Text("Cancelling...")
                            .font(.system(size: 9))
                            .foregroundColor(.orange)
                    }
                } else if inferenceService.hasLoadingError {
                    // Display error indicator when loading fails
                    Button(action: { showErrorDetails = true }) {
                        HStack(spacing: 4) {
                            Image(systemName: "exclamationmark.circle")
                                .foregroundColor(.red)
                                .font(.caption)
                            
                            Text("Load Error")
                                .font(.system(size: 9))
                                .foregroundColor(.red)
                        }
                    }
                    .buttonStyle(PlainButtonStyle())
                    .popover(isPresented: $showErrorDetails) {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Model Loading Error")
                                .font(.headline)
                                .padding(.bottom, 4)
                            
                            // Show all errors from the current loading attempt
                            if !inferenceService.currentLoadingErrors.isEmpty {
                                VStack(alignment: .leading, spacing: 8) {
                                    ForEach(inferenceService.currentLoadingErrors, id: \.self) { error in
                                        Text(error)
                                            .font(.subheadline)
                                            .foregroundColor(.red)
                                    }
                                }
                            } else if let errorMessage = inferenceService.lastLoadingError {
                                Text(errorMessage)
                                    .font(.subheadline)
                                    .foregroundColor(.red)
                            } else {
                                Text("An unknown error occurred while loading the model.")
                                    .font(.subheadline)
                                    .foregroundColor(.red)
                            }
                            
                            Divider()
                            
                            Text("Suggested Actions:")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            VStack(alignment: .leading, spacing: 8) {
                                Text("• Verify model files are complete")
                                Text("• Try reloading the model")
                                Text("• Check available storage space")
                                Text("• Reinstall the model if issues persist")
                            }
                            .font(.caption)
                            .foregroundColor(.secondary)
                            
                            HStack {
                                Spacer()
                                Button("Close") {
                                    showErrorDetails = false
                                }
                                .padding(.top, 8)
                            }
                        }
                        .padding()
                        .frame(width: 300)
                    }
                } else if isModelLoading {
                    // Use our updated ModelLoadingView
                    ModelLoadingView()
                        .frame(width: 200)
                }
                
                Button(action: onReloadModel) {
                    Image(systemName: "arrow.triangle.2.circlepath").foregroundColor(.blue)
                }
                .disabled(isCancellingModel) // Disable the button while cancelling
            } 
            // When no model is available, show the warning state
            else {
                // Fallback view with button to open model management
                Button(action: { showModelManagementView() }) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill").foregroundColor(.orange)
                        Text("No Model Selected").font(.caption).foregroundColor(.secondary)
                    }
                }
                .buttonStyle(PlainButtonStyle())
                
                Spacer()
                
                // Add Download Llama button
                Button(action: {
                    // Dismiss any active keyboard first to prevent it showing
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                    
                    showModelManagementView()
                    // Signal to download the default model
                    NotificationCenter.default.post(name: Notification.Name("DownloadDefaultModel"), object: nil)
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.down.circle")
                        Text("Download Llama 3.2")
                    }
                    .font(.caption)
                }
                .buttonStyle(PlainButtonStyle())
                .foregroundColor(.blue)
                .padding(.horizontal, 4)
            }
        }
        .padding(.horizontal).padding(.vertical, 4).background(Color(.systemGray6))
        // Force this view to render regardless of model status
        .onAppear {
            print("DEBUG: ModelIndicatorView appeared, model: \(modelService.getSelectedModel()?.name ?? "none")")
        }
    }
}

struct SettingsView: View {
    var body: some View { Text("Settings") }
}
