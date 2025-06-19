// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// HomeView.swift

// HomeView.swift
// Modified HomeView.swift
import SwiftUI

struct HomeView: View {
    @State private var chats: [Chat] = []
    @State private var selectedChatId: String?
    // Add a secondary chat ID to track what's actually loaded
    @State private var loadedChatId: String?
    @State private var showingAbout = false // Add this state
    @State private var showingModelManagement = false // Add state for model management
    @State private var errorMessage: String?
    @State private var showingError = false
    @StateObject private var modelService = ModelService.shared
    @State private var selectedModelIndex: Int = 0
    @State private var isShowingAlert = false
    @State private var alertTitle: String = ""
    @State private var alertMessage: String = ""
    @State private var isSelectingModel = false
    // Add a timer to periodically check if views are in sync
    @State private var selectionCheckTimer: Timer? = nil
    // Add state for column visibility with a default value
    @State private var columnVisibility: NavigationSplitViewVisibility = .automatic

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            sidebarContent
        } detail: {
            detailContent
        }
        .onAppear {
            print("DEBUG: HomeView appeared")
            loadChats()
            
            // Load saved sidebar state
            loadSidebarState()
            
            // Check if we need to show model management view
            checkModelAvailability()
            
            // Start a timer to check selection sync periodically
            startSelectionCheckTimer()
        }
        .onDisappear {
            // Clean up timer
            selectionCheckTimer?.invalidate()
            selectionCheckTimer = nil
        }
        .onChange(of: columnVisibility) { oldValue, newValue in
            // Save sidebar state when it changes
            saveSidebarState()
        }
        .alert("Error", isPresented: $showingError, actions: {
            Button("OK", role: .cancel) {}
        }, message: {
            Text(errorMessage ?? "An unknown error occurred")
        })
        .alert(alertTitle, isPresented: $isShowingAlert, actions: {
            Button("OK", role: .cancel) {}
        }, message: {
            Text(alertMessage)
        })
    }
    
    // MARK: - Sidebar State Management
    
    private func saveSidebarState() {
        // Convert NavigationSplitViewVisibility to Int for storage
        let visibilityValue: Int
        switch columnVisibility {
        case .detailOnly:
            visibilityValue = 0
        case .automatic:
            visibilityValue = 1
        case .all:
            visibilityValue = 2
        default:
            visibilityValue = 1 // Default to automatic for unknown cases
        }
        
        // Save to UserDefaults
        UserDefaults.standard.set(visibilityValue, forKey: "sidebarVisibility")
        print("DEBUG: Saved sidebar visibility state: \(visibilityValue)")
    }
    
    private func loadSidebarState() {
        // Default to automatic (1) if no value is stored
        let visibilityValue = UserDefaults.standard.integer(forKey: "sidebarVisibility")
        
        // Convert Int back to NavigationSplitViewVisibility
        switch visibilityValue {
        case 0:
            columnVisibility = .detailOnly
        case 1:
            columnVisibility = .automatic
        case 2:
            columnVisibility = .all
        default:
            columnVisibility = .automatic
        }
        
        print("DEBUG: Loaded sidebar visibility state: \(visibilityValue)")
    }
    
    // MARK: - Extracted View Components
    
    private var sidebarContent: some View {
        List(selection: $selectedChatId) {
            ForEach(chats) { chat in
                chatRow(for: chat)
            }
        }
        .navigationTitle("Chats")
        .toolbar {
            sidebarToolbarItems
        }
        .sheet(isPresented: $showingAbout) { // Show About view
            AboutView()
        }
        // Add sheet for model management
        .sheet(isPresented: $showingModelManagement, onDismiss: {
            // Inform the coordinator that the view has been dismissed
            ModelManagementCoordinator.didDismiss()
            
            // Check if we still need a model after dismissal
            let needModel = modelService.getSelectedModel() == nil || 
                          (modelService.getSelectedModel() != nil && 
                           !modelService.verifyModelFiles(modelId: modelService.getSelectedModel()!.id))
            
            if needModel {
                print("DEBUG: Model still required after dismiss attempt from HomeView")
                // Add delay before attempting to show again to avoid presentation conflicts
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    Task { @MainActor in
                        if ModelManagementCoordinator.canPresent() {
                            print("DEBUG: Re-showing ModelManagementView after failed dismiss from HomeView")
                            showingModelManagement = true
                        }
                    }
                }
            } else {
                print("DEBUG: Model verification successful, HomeView ModelManagementView dismissed")
            }
        }) {
            ModelManagementView()
                .id(UUID()) // Force a fresh instance each time
                .interactiveDismissDisabled(modelService.getSelectedModel() == nil || 
                                           (modelService.getSelectedModel() != nil && 
                                            !modelService.verifyModelFiles(modelId: modelService.getSelectedModel()!.id)))
                .onAppear {
                    print("DEBUG: ModelManagementView appeared from HomeView")
                }
                .onDisappear {
                    print("DEBUG: ModelManagementView disappeared from HomeView")
                }
        }
        .onChange(of: selectedChatId) { oldValue, newValue in
            print("DEBUG: selectedChatId changed from \(oldValue ?? "nil") to \(newValue ?? "nil")")
            if let chatId = newValue {
                syncLoadedChat(chatId)
            }
        }
    }
    
    private func chatRow(for chat: Chat) -> some View {
        NavigationLink(value: chat.id) {
            VStack(alignment: .leading) {
                Text(chat.title)
                    .font(.headline)
                Text(chat.lastMessagePreview)
                    .font(.subheadline)
                    .foregroundColor(.gray)
                    .lineLimit(1)
            }
            // Add a tap gesture to ensure selection works even if SwiftUI doesn't update
            .onTapGesture {
                print("DEBUG: Direct tap on chat item with ID: \(chat.id)")
                forceSelectChat(chat)
            }
        }
        .contextMenu {
            Button(role: .destructive, action: {
                deleteChat(chat)
            }) {
                Label("Delete", systemImage: "trash")
            }
        }
        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
            Button(role: .destructive) {
                deleteChat(chat)
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }
    
    private var sidebarToolbarItems: some ToolbarContent {
        Group {
            ToolbarItem(placement: .principal) {
                Text("Chats")
                    .font(.headline)
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: {
                    // Direct call to newChat without any checks
                    newChat()
                }) {
                    Image(systemName: "plus.circle.fill")
                        .foregroundColor(.blue)
                        .font(.system(size: 20))
                }
            }
            ToolbarItem(placement: .navigationBarLeading) { // Add About button
                Button(action: { showingAbout = true }) {
                    Image(systemName: "info.circle")
                }
            }
            // Add model management button
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: { showModelManagementView() }) {
                    #if os(iOS)
                    // Use the same icon as on macOS
                    Image(systemName: "cube.box")
                    #else
                    Image(systemName: "cube.box")
                    #endif
                }
            }
        }
    }
    
    private var detailContent: some View {
        Group {
            if let loadedChatId = loadedChatId,
               let chat = chats.first(where: { $0.id == loadedChatId }) {
                ChatView(chat: chat)
                    .id(loadedChatId)
                    .onAppear {
                        print("DEBUG: Creating ChatView for chat ID: \(loadedChatId), title: \(chat.title), messages: \(chat.messages.count)")
                    }
            } else {
                emptyDetailView
            }
        }
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: {
                    // Direct call to newChat without any checks
                    newChat()
                }) {
                    Image(systemName: "plus.circle.fill")
                        .foregroundColor(.blue)
                        .font(.system(size: 20))
                }
            }
        }
    }
    
    private var emptyDetailView: some View {
        VStack(spacing: 20) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            Text("Select a chat or create a new one")
                .font(.headline)
                .foregroundColor(.gray)
            
            // Check if a model is available
            if modelService.getSelectedModel() == nil {
                // No model selected
                VStack(spacing: 12) {
                    Text("No model selected")
                        .font(.subheadline)
                        .foregroundColor(.orange)
                    
                    Button(action: {
                        showingModelManagement = true
                    }) {
                        Text("Select a model")
                            .font(.subheadline)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                }
            } else if let selectedModel = modelService.getSelectedModel() {
                // Model selected, but check if it's downloaded and has all files
                if !selectedModel.isDownloaded {
                    // Model not downloaded
                    VStack(spacing: 12) {
                        Text("Selected model not downloaded")
                            .font(.subheadline)
                            .foregroundColor(.orange)
                        
                        Button(action: {
                            showingModelManagement = true
                            // Notify ModelManagementView to start downloading the model
                            NotificationCenter.default.post(name: Notification.Name("DownloadSelectedModel"), object: selectedModel.id)
                        }) {
                            Text("Download model")
                                .font(.subheadline)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                    }
                } else if modelService.hasIncompleteDownload(modelId: selectedModel.id) {
                    // Model has incomplete files
                    VStack(spacing: 12) {
                        HStack {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.orange)
                            Text("Model has missing files")
                                .font(.subheadline)
                                .foregroundColor(.orange)
                        }
                        
                        Button(action: {
                            showingModelManagement = true
                        }) {
                            Text("Verify and repair")
                                .font(.subheadline)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(Color.orange)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                    }
                } else if !modelService.verifyModelFiles(modelId: selectedModel.id) {
                    // Model verification failed
                    VStack(spacing: 12) {
                        Text("Model verification failed")
                            .font(.subheadline)
                            .foregroundColor(.orange)
                        
                        Button(action: {
                            showingModelManagement = true
                        }) {
                            Text("Manage models")
                                .font(.subheadline)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                    }
                }
            }
            
            // Debug info - show what chat IDs we have
            #if DEBUG
            debugInfoView
            #endif
            
            Button(action: {
                // Direct call to newChat without any checks
                newChat()
            }) {
                Label("New Chat", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
            
            // Add a button to retry loading the selected chat
            if selectedChatId != nil && selectedChatId != loadedChatId {
                Button("Retry Loading Chat") {
                    if let chatId = selectedChatId {
                        syncLoadedChat(chatId)
                    }
                }
                .padding(.top, 10)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(UIColor.systemGroupedBackground))
        .onAppear {
            print("DEBUG: Empty selection view appeared. selectedChatId: \(selectedChatId ?? "nil"), chats count: \(chats.count)")
            
            // If there's a selected chat but no loaded chat, try to sync them
            if selectedChatId != nil && loadedChatId == nil {
                if let chatId = selectedChatId {
                    syncLoadedChat(chatId)
                }
            }
        }
    }
    
    private var debugInfoView: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Debug Info:")
                .font(.caption)
                .foregroundColor(.secondary)
            Text("Selected ID: \(selectedChatId ?? "nil")")
                .font(.caption)
                .foregroundColor(.secondary)
            Text("Loaded ID: \(loadedChatId ?? "nil")")
                .font(.caption)
                .foregroundColor(.secondary)
            Text("Chat count: \(chats.count)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(8)
        .padding(.top, 20)
    }

    // MARK: - Functions
    
    private func loadChats() {
        print("DEBUG: Loading chats")
        chats = ChatService.shared.loadChats()
        print("DEBUG: Loaded \(chats.count) chats")
        
        if chats.isEmpty {
            print("DEBUG: No chats found, creating a new one")
            newChat()
        } else if selectedChatId == nil {
            print("DEBUG: Setting selectedChatId to first chat: \(chats.first?.id ?? "nil")")
            let firstChatId = chats.first?.id
            selectedChatId = firstChatId
            loadedChatId = firstChatId  // Also set the loadedChatId
        }
    }

    func newChat() {
        print("DEBUG: Creating new chat")
        
        // Get the model ID to use - even if not downloaded
        let modelId: String
        
        // First try to use the selected model, even if not downloaded
        if let selectedModel = modelService.selectedModel {
            modelId = selectedModel.id
            print("DEBUG: Using selected model for new chat: \(selectedModel.name)")
        } 
        // If no selected model, use the first available model (likely the default Llama 3.2 1B)
        else if let firstModel = modelService.getAvailableModels().first {
            modelId = firstModel.id
            print("DEBUG: Using first available model for new chat: \(firstModel.name)")
            
            // Select this model in the service
            Task { @MainActor in
                modelService.selectModel(firstModel)
            }
        } 
        // Fallback to hardcoded ID in the extremely unlikely case no models are available
        else {
            modelId = "llama-3.2-1b" // Default model ID
            print("DEBUG: Using hardcoded default model ID: \(modelId)")
        }
        
        // Create a new chat with the selected/default model ID
        let newChat = ChatService.shared.createChat(modelId: modelId)
        print("DEBUG: Created new chat with ID: \(newChat.id), using model: \(modelId)")
        
        // Update the UI
        chats.insert(newChat, at: 0)
        selectedChatId = newChat.id
        loadedChatId = newChat.id  // Also set loadedChatId
        print("DEBUG: Set selectedChatId to new chat: \(newChat.id)")
        
        // If no models are downloaded, show model management view
        if modelService.downloadedModels.isEmpty {
            print("DEBUG: No models downloaded - showing model management view after chat creation")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.showModelManagementView()
            }
        }
    }
    
    func deleteChat(_ chat: Chat) {
        // Delete the chat from the service
        Task {
            do {
                try await ChatService.shared.deleteChat(chat)
                
                // Remove from the UI
                if let index = chats.firstIndex(where: { $0.id == chat.id }) {
                    chats.remove(at: index)
                }
                
                // If the deleted chat was selected, select another one
                if selectedChatId == chat.id {
                    let newSelectedId = chats.first?.id
                    selectedChatId = newSelectedId
                    loadedChatId = newSelectedId
                    print("DEBUG: Deleted selected chat, now selected: \(selectedChatId ?? "nil")")
                }
            } catch {
                print("Error deleting chat: \(error)")
            }
        }
    }
    
    // Helper method to force selection of a chat
    private func forceSelectChat(_ chat: Chat) {
        print("DEBUG: Force selecting chat with ID: \(chat.id)")
        selectedChatId = chat.id
        syncLoadedChat(chat.id)
    }
    
    // Helper method to ensure loadedChatId and selectedChatId stay in sync
    private func syncLoadedChat(_ chatId: String) {
        print("DEBUG: Syncing loaded chat to ID: \(chatId)")
        
        // Check if the chat exists
        if let _ = chats.first(where: { $0.id == chatId }) {
            // Use a slight delay before updating loadedChatId to allow SwiftUI to update
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                loadedChatId = chatId
                print("DEBUG: Updated loadedChatId to: \(chatId)")
            }
        } else {
            print("DEBUG: Cannot sync to chat ID \(chatId) - chat not found")
        }
    }
    
    // Start a timer to periodically check if selectedChatId and loadedChatId are in sync
    private func startSelectionCheckTimer() {
        selectionCheckTimer?.invalidate()
        
        selectionCheckTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            // Check if they're out of sync
            if let selectedId = selectedChatId, selectedId != loadedChatId {
                print("DEBUG: Selection check - out of sync. selectedChatId: \(selectedId), loadedChatId: \(loadedChatId ?? "nil")")
                
                // Try to sync them
                if let _ = chats.first(where: { $0.id == selectedId }) {
                    DispatchQueue.main.async {
                        syncLoadedChat(selectedId)
                    }
                }
            }
        }
    }

    private func selectModel() {
        guard selectedModelIndex < modelService.getAvailableModels().count else {
            print("Error: Selected model index out of bounds")
            return
        }
        
        let model = modelService.getAvailableModels()[selectedModelIndex]
        
        // Check if model is already selected to avoid unnecessary updates
        if modelService.isModelSelected(model) {
            print("Model \(model.name) is already selected")
            return
        }
        
        // Check if model is downloaded
        if !model.isDownloaded {
            print("Cannot select model \(model.name) because it is not downloaded")
            // Show an alert instead of trying to select
            isShowingAlert = true
            alertTitle = "Model Not Downloaded"
            alertMessage = "Please download the model before selecting it."
            return
        }
        
        // Use Task to ensure we're on the main actor
        Task { @MainActor in
            // Prevent multiple UI updates at once
            if !isSelectingModel {
                isSelectingModel = true
                
                do {
                    // Try to load the model for inference first
                    try await modelService.loadModelForInference(model)
                    
                    // If successful, select the model (we're already on MainActor)
                    modelService.selectModel(model)
                    print("Successfully selected and loaded model: \(model.name)")
                } catch {
                    print("Error loading model for inference: \(error)")
                    // Show error alert
                    self.isShowingAlert = true
                    self.alertTitle = "Error Selecting Model"
                    self.alertMessage = "Failed to load the model: \(error.localizedDescription)"
                }
                
                isSelectingModel = false
            } else {
                print("Model selection already in progress, ignoring request")
            }
        }
    }

    // MARK: - Model Management
    
    // Coordinator for model management view presentation across the app
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
    
    private func checkModelAvailability() {
        print("DEBUG: Checking model availability in HomeView")
        
        // Check if a model is selected
        guard let selectedModel = modelService.getSelectedModel() else {
            print("DEBUG: No model selected, showing model management view from HomeView")
            showModelManagementView()
            return
        }
        
        // Check if the selected model is properly downloaded and verified
        let isModelValid = modelService.verifyModelFiles(modelId: selectedModel.id)
        if !isModelValid {
            print("DEBUG: Selected model verification failed, showing model management view from HomeView")
            showModelManagementView()
            return
        }
        
        print("DEBUG: Model check passed for \(selectedModel.name) in HomeView")
    }
    
    private func showModelManagementView() {
        // Use Task to safely dispatch UI updates
        Task { @MainActor in
            if !showingModelManagement {
                if ModelManagementCoordinator.canPresent() {
                    print("DEBUG: Showing ModelManagementView from HomeView")
                    showingModelManagement = true
                } else {
                    print("DEBUG: ModelManagementView presentation prevented by coordinator in HomeView")
                }
            } else {
                print("DEBUG: ModelManagementView already showing in HomeView, ignoring tap")
            }
        }
    }
}
