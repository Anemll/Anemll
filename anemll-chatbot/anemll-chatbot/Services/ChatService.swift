// ChatService.swift
// Copyright (c) 2025 Anemll
// Licensed under the MIT License

import Foundation
import Combine
import AnemllCore  // Import AnemllCore for Tokenizer.ChatMessage

// Add a struct to pass inference metrics along with the response text
struct ChatResponse {
    let text: String
    let tokensPerSecond: Double
    let tokenCount: Int
    let windowShifts: Int? // New field to track window shifts
    let isComplete: Bool // Whether this is the final result
}

class ChatService: ObservableObject {
    @Published var messages: [Message] = []
    @Published var isGenerating: Bool = false
    @Published var modelLoadingStatus: String?
    
    private var modelService = ModelService.shared
    private var cancellables = Set<AnyCancellable>()
    private var isModelLoaded = false
    
    // Singleton instance for easy access
    static let shared = ChatService()
    
    // File management properties
    private let fileManager = FileManager.default
    private let documentDirectory: URL
    
    // Private initializer to set up the document directory
    private init() {
        documentDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        
        // Add a welcome message
        let welcomeMessage = Message(
            text: "Hello! I'm Anemll, powered by Llama 3.2. How can I help you today?",
            isUser: false
        )
        messages.append(welcomeMessage)
        
        // Listen for model loading events
        setupModelLoadingObservers()
    }
    
    private func setupModelLoadingObservers() {
        NotificationCenter.default.addObserver(
            forName: Notification.Name("ModelLoadingStarted"),
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isModelLoaded = false
            self?.modelLoadingStatus = "Loading model..."
        }
        
        NotificationCenter.default.addObserver(
            forName: Notification.Name("ModelLoadingCompleted"),
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isModelLoaded = true
            self?.modelLoadingStatus = nil
        }
        
        NotificationCenter.default.addObserver(
            forName: Notification.Name("ModelLoadingFailed"),
            object: nil,
            queue: .main
        ) { [weak self] notification in
            self?.isModelLoaded = false
            if let error = notification.userInfo?["error"] as? String {
                self?.modelLoadingStatus = "CS.1 Error loading model: \(error)"
            } else {
                self?.modelLoadingStatus = "CS.2 Error loading model"
            }
        }
        
        NotificationCenter.default.addObserver(
            forName: Notification.Name("ModelLoadingInterrupted"),
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isModelLoaded = false
            self?.modelLoadingStatus = "Model loading interrupted"
        }
    }
    
    /// Saves a chat to a JSON file in the document directory
    func saveChat(_ chat: Chat) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        do {
            let data = try encoder.encode(chat)
            let fileURL = documentDirectory.appendingPathComponent("\(chat.id).json")
            try data.write(to: fileURL)
        } catch {
            print("Error saving chat: \(error)")
        }
    }
    
    /// Loads all chats from JSON files in the document directory
    func loadChats() -> [Chat] {
        do {
            let files = try fileManager.contentsOfDirectory(at: documentDirectory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            var chats: [Chat] = []
            for file in files where file.pathExtension == "json" {
                let data = try Data(contentsOf: file)
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let chat = try decoder.decode(Chat.self, from: data)
                chats.append(chat)
            }
            // Sort chats by creation date (newest first)
            return chats.sorted(by: { $0.createdAt > $1.createdAt })
        } catch {
            print("Error loading chats: \(error)")
            return []
        }
    }
    
    /// Deletes a chat by removing its JSON file
    func deleteChat(_ chat: Chat) async throws {
        let fileURL = documentDirectory.appendingPathComponent("\(chat.id).json")
        try fileManager.removeItem(at: fileURL)
    }
    
    /// Creates a new chat with the specified model ID
    func createChat(modelId: String) -> Chat {
        let chat = Chat(modelId: modelId)
        saveChat(chat)
        
        // Initialize conversation state for the new chat
        Task { @MainActor in
            InferenceService.shared.initializeConversationState(for: chat.id)
        }
        
        return chat
    }
    
    /// Generates a response for a user message using the InferenceService
    /// Returns an AsyncStream of response chunks with metrics for real-time display
    func generateResponse(for chat: Chat, userMessage: String, allowLongGeneration: Bool = true) async throws -> AsyncStream<ChatResponse> {
        print("DEBUG - ChatService.generateResponse called with message: \"\(userMessage)\"")
        print("DEBUG - Chat ID: \(chat.id), Model ID: \(chat.modelId)")
        print("DEBUG - Long generation: \(allowLongGeneration ? "enabled" : "disabled")")
        
        // Ensure the model is loaded
        let inferenceService = await InferenceService.shared
        
        // Check if the requested model ID matches the currently loaded model
        let requestedModelId = chat.modelId
        let currentModelId = await inferenceService.currentModel
        
        // If there's a model mismatch, use the current model or handle appropriately
        let effectiveModelId: String
        
        if currentModelId != nil && currentModelId != requestedModelId {
            // Model mismatch detected
            print("DEBUG - Model ID mismatch detected. Chat requests: \(requestedModelId), Currently loaded: \(currentModelId!)")
            
            if await inferenceService.isModelLoaded {
                // Use currently loaded model instead
                print("DEBUG - Using currently loaded model: \(currentModelId!) instead of requested: \(requestedModelId)")
                effectiveModelId = currentModelId!
            } else {
                // No model is loaded, use requested model
                print("DEBUG - No model loaded, using requested model: \(requestedModelId)")
                effectiveModelId = requestedModelId
            }
        } else {
            // No mismatch, use requested model
            effectiveModelId = requestedModelId
        }
        
        // Generate the response using the InferenceService with the effective model ID
        print("DEBUG - Calling inferenceService.inferStream with effectiveModelId: \(effectiveModelId)")
        let resultStream = try await inferenceService.inferStream(
            modelId: effectiveModelId, 
            input: userMessage,
            allowLongGeneration: allowLongGeneration
        )
        
        // Convert InferenceResult stream to ChatResponse stream
        return AsyncStream<ChatResponse> { continuation in
            Task {
                for await result in resultStream {
                    continuation.yield(ChatResponse(
                        text: result.text,
                        tokensPerSecond: result.tokensPerSecond,
                        tokenCount: result.tokenCount,
                        windowShifts: result.windowShifts,
                        isComplete: result.isComplete
                    ))
                }
                continuation.finish()
            }
        }
    }
    
    /// Generates a response with a system prompt for more control
    func generateResponseWithSystemPrompt(
        for chat: Chat, 
        systemPrompt: String, 
        userMessage: String,
        allowLongGeneration: Bool = true
    ) async throws -> AsyncStream<ChatResponse> {
        print("DEBUG - ChatService.generateResponseWithSystemPrompt called")
        print("DEBUG - System prompt: \"\(systemPrompt)\"")
        print("DEBUG - User message: \"\(userMessage)\"")
        print("DEBUG - Chat history messages count: \(chat.messages.count)")
        print("DEBUG - Long generation: \(allowLongGeneration ? "enabled" : "disabled")")
        
        // Ensure the model is loaded
        let inferenceService = await InferenceService.shared
        
        // Check if the requested model ID matches the currently loaded model
        let requestedModelId = chat.modelId
        let currentModelId = await inferenceService.currentModel
        
        // If there's a model mismatch, use the current model or handle appropriately
        let effectiveModelId: String
        
        if currentModelId != nil && currentModelId != requestedModelId {
            // Model mismatch detected
            print("DEBUG - Model ID mismatch detected. Chat requests: \(requestedModelId), Currently loaded: \(currentModelId!)")
            
            if await inferenceService.isModelLoaded {
                // Use currently loaded model instead
                print("DEBUG - Using currently loaded model: \(currentModelId!) instead of requested: \(requestedModelId)")
                effectiveModelId = currentModelId!
            } else {
                // No model is loaded, use requested model
                print("DEBUG - No model loaded, using requested model: \(requestedModelId)")
                effectiveModelId = requestedModelId
            }
        } else {
            // No mismatch, use requested model
            effectiveModelId = requestedModelId
        }
        
        // Convert full chat history to Tokenizer.ChatMessage format
        var chatMessages: [Tokenizer.ChatMessage] = []
        
        // First add system prompt if provided
        if !systemPrompt.isEmpty {
            chatMessages.append(.assistant(systemPrompt))
        }
        
        // Add all previous messages from the chat history
        for message in chat.messages {
            if message.text.isEmpty { continue } // skip empty messages!
            if message.isUser {
                chatMessages.append(.user(message.text))
            } else {
                // Important: AI responses must be tokenized as assistant messages
                chatMessages.append(.assistant(message.text))
            }
        }
        
        // Add the current user message if it's not already included
        // This handles the case where the UI might add the message to chat.messages before calling this method
        if !chat.messages.contains(where: { $0.isUser && $0.text == userMessage }) {
            chatMessages.append(.user(userMessage))
        }
        
        print("DEBUG - Prepared chat history with \(chatMessages.count) messages for inference")
        
        // Generate the response using the InferenceService with system prompt and full history
        print("DEBUG - Calling inferenceService.inferWithSystemPrompt with effectiveModelId: \(effectiveModelId)")
        let resultStream = try await inferenceService.inferWithSystemPrompt(
            modelId: effectiveModelId,
            systemPrompt: systemPrompt,
            userInput: userMessage,
            chatHistory: chatMessages,  // Pass the complete conversation history
            chatId: chat.id,  // Pass the chat ID for state tracking
            allowLongGeneration: allowLongGeneration
        )
        
        // Convert InferenceResult stream to ChatResponse stream
        return AsyncStream<ChatResponse> { continuation in
            Task {
                for await result in resultStream {
                    continuation.yield(ChatResponse(
                        text: result.text,
                        tokensPerSecond: result.tokensPerSecond,
                        tokenCount: result.tokenCount,
                        windowShifts: result.windowShifts,
                        isComplete: result.isComplete
                    ))
                }
                continuation.finish()
            }
        }
    }
    
    func sendMessage(_ content: String) {
        // Add user message
        let userMessage = Message(
            text: content,
            isUser: true
        )
        messages.append(userMessage)
        
        // Check if model is loaded
        if !isModelLoaded {
            // If model isn't loaded, start loading it
            if let selectedModel = modelService.getSelectedModel() {
                if selectedModel.isDownloaded {
                    // Trigger model loading if not already loading
                    if modelLoadingStatus == nil {
                        modelService.preloadSelectedModel()
                    }
                    
                    // Add a system message indicating model is loading
                    let loadingMessage = Message(
                        text: "Loading the model. Please wait...",
                        isUser: false
                    )
                    messages.append(loadingMessage)
                    return
                } else {
                    // Model isn't downloaded
                    let errorMessage = Message(
                        text: "The selected model isn't downloaded. Please download it in Settings.",
                        isUser: false
                    )
                    messages.append(errorMessage)
                    return
                }
            } else {
                // No model selected
                let errorMessage = Message(
                    text: "No model is selected. Please select a model in Settings.",
                    isUser: false
                )
                messages.append(errorMessage)
                return
            }
        }
        
        // Generate response
        generateResponse(to: content)
    }
    
    private func generateResponse(to userMessage: String) {
        isGenerating = true
        
        // Simulate thinking time
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self = self else { return }
            
            // Create a placeholder for the assistant's response
            let placeholderMessage = Message(
                text: "",
                isUser: false
            )
            self.messages.append(placeholderMessage)
            
            // Simulate typing with word-by-word updates
            self.simulateTypingResponse(message: placeholderMessage, userMessage: userMessage)
        }
    }
    
    private func simulateTypingResponse(message: Message, userMessage: String) {
        // This is a placeholder for actual model inference
        // In a real app, you would call your ML model here
        
        // Generate a simple response based on the user's message
        var response = ""
        if userMessage.lowercased().contains("hello") || userMessage.lowercased().contains("hi") {
            response = "Hello! How can I help you today?"
        } else if userMessage.lowercased().contains("how are you") {
            response = "I'm just a language model, but I'm functioning well. Thanks for asking! How can I assist you?"
        } else if userMessage.lowercased().contains("name") {
            response = "I'm Anemll, a language model based on Llama 3.2. It's nice to meet you!"
        } else if userMessage.lowercased().contains("thank") {
            response = "You're welcome! Feel free to ask if you need anything else."
        } else {
            response = "I understand you're saying: \"\(userMessage)\". As a language model, I'm here to help answer questions and provide information. What would you like to know more about?"
        }
        
        // Split the response into words for the typing animation
        let words = response.split(separator: " ").map(String.init)
        
        // Simulate typing word by word
        var currentResponse = ""
        var wordIndex = 0
        
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] timer in
            guard let self = self else {
                timer.invalidate()
                return
            }
            
            if wordIndex < words.count {
                if wordIndex > 0 {
                    currentResponse += " "
                }
                currentResponse += words[wordIndex]
                wordIndex += 1
                
                // Update the message
                if let index = self.messages.firstIndex(where: { $0.id == message.id }) {
                    self.messages[index].text = currentResponse
                }
            } else {
                // Finished typing
                timer.invalidate()
                self.isGenerating = false
            }
        }
    }
    
    /// Gets a chat by ID
    private func getChat(id: String) -> Chat? {
        let fileURL = documentDirectory.appendingPathComponent("\(id).json")
        
        guard fileManager.fileExists(atPath: fileURL.path) else {
            return nil
        }
        
        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(Chat.self, from: data)
        } catch {
            print("Error loading chat \(id): \(error)")
            return nil
        }
    }
    
    /// Loads a chat from storage by ID
    func loadChat(id: String) -> Chat? {
        guard let chat = getChat(id: id) else {
            return nil
        }
        
        // Initialize conversation state for the loaded chat
        Task { @MainActor in
            InferenceService.shared.initializeConversationState(for: chat.id)
            
            // Update token count based on message count (approximate)
            // In a real implementation, we would track actual token counts
            let approximateTokenCount = chat.messages.count * 20 // Rough estimate
            InferenceService.shared.updateConversationState(tokenCount: approximateTokenCount, chatId: chat.id)
        }
        
        return chat
    }
}
