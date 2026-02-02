//
//  ChatViewModel.swift
//  ANEMLLChat
//
//  ViewModel for chat functionality
//

import Foundation
import SwiftUI
import Observation

/// Main view model for chat functionality
@Observable
@MainActor
final class ChatViewModel {
    // MARK: - Published State

    /// All conversations
    var conversations: [Conversation] = []

    /// Currently selected conversation
    var currentConversation: Conversation?

    /// Text in the input field
    var inputText: String = ""

    /// Whether generation is in progress
    var isGenerating: Bool = false

    /// Currently streaming message content
    var streamingContent: String = ""

    /// Error message to display
    var errorMessage: String?

    /// Window shifts during current generation
    var currentWindowShifts: Int = 0

    /// Current tokens per second
    var currentTokensPerSecond: Double = 0

    /// Current history tokens during streaming (input + output so far)
    var currentHistoryTokens: Int = 0

    // MARK: - Dependencies

    private let inferenceService = InferenceService.shared

    // MARK: - Initialization

    init() {
        Task {
            await loadConversations()
        }
    }

    // MARK: - Conversation Management

    /// Load all conversations from storage
    func loadConversations() async {
        do {
            conversations = try await StorageService.shared.loadConversations()
            logInfo("Loaded \(conversations.count) conversations", category: .app)

            if currentConversation == nil {
                if let first = conversations.first {
                    currentConversation = first
                } else {
                    newConversation()
                }
            }
        } catch {
            logError("Failed to load conversations: \(error)", category: .storage)
            errorMessage = "Failed to load conversations"

            if currentConversation == nil {
                newConversation()
            }
        }
    }

    /// Create a new conversation
    func newConversation() {
        let conversation = Conversation()
        conversations.insert(conversation, at: 0)
        currentConversation = conversation

        Task {
            try? await StorageService.shared.saveConversation(conversation)
        }

        logDebug("Created new conversation", category: .app)
    }

    /// Select a conversation
    func selectConversation(_ conversation: Conversation) {
        currentConversation = conversation
    }

    /// Delete a conversation
    func deleteConversation(_ conversation: Conversation) {
        conversations.removeAll { $0.id == conversation.id }

        if currentConversation?.id == conversation.id {
            currentConversation = conversations.first
        }

        Task {
            try? await StorageService.shared.deleteConversation(conversation.id)
        }

        logDebug("Deleted conversation: \(conversation.id)", category: .app)
    }

    /// Delete conversation at index
    func deleteConversation(at indexSet: IndexSet) {
        for index in indexSet {
            let conversation = conversations[index]
            deleteConversation(conversation)
        }
    }

    // MARK: - Message Sending

    /// Send a message and generate a response
    func sendMessage() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        guard !isGenerating else { return }

        // Ensure we have a conversation
        if currentConversation == nil {
            newConversation()
        }

        guard var conversation = currentConversation else { return }

        // Clear input
        inputText = ""

        // Add user message
        let userMessage = ChatMessage.user(text)
        conversation.addMessage(userMessage)

        // Add placeholder assistant message
        let assistantMessage = ChatMessage.assistant("", isComplete: false)
        conversation.addMessage(assistantMessage)

        currentConversation = conversation
        updateConversationInList(conversation)

        // Start generation
        isGenerating = true
        streamingContent = ""
        currentWindowShifts = 0
        currentTokensPerSecond = 0
        currentHistoryTokens = 0
        errorMessage = nil

        do {
            // Get messages for context (exclude the empty assistant message)
            let contextMessages = conversation.messages.filter {
                $0.role != .assistant || !$0.content.isEmpty
            }

            let result = try await inferenceService.generateResponse(
                for: contextMessages,
                onToken: { [weak self] token in
                    Task { @MainActor in
                        guard let self = self else { return }
                        self.streamingContent += token
                        self.updateStreamingMessage()
                    }
                },
                onWindowShift: { [weak self] in
                    Task { @MainActor in
                        guard let self = self else { return }
                        self.currentWindowShifts += 1
                    }
                },
                onHistoryUpdate: { [weak self] historyTokens in
                    Task { @MainActor in
                        guard let self = self else { return }
                        self.currentHistoryTokens = historyTokens
                        self.updateStreamingMessage()
                    }
                }
            )

            // Update final message
            conversation.updateLastAssistantMessage(
                content: result.text,
                tokensPerSecond: result.tokensPerSecond,
                tokenCount: result.tokenCount,
                windowShifts: result.windowShifts,
                prefillTime: result.prefillTime,
                prefillTokens: result.prefillTokens,
                historyTokens: result.historyTokens,
                isComplete: true,
                wasCancelled: result.wasCancelled,
                stopReason: result.stopReason
            )

            currentConversation = conversation
            updateConversationInList(conversation)

            // Save
            try? await StorageService.shared.saveConversation(conversation)

            logInfo("Message generated: \(result.tokenCount) tokens", category: .app)

        } catch {
            logError("Generation failed: \(error)", category: .inference)
            errorMessage = error.localizedDescription

            // Remove the empty assistant message on error
            conversation.messages.removeLast()
            currentConversation = conversation
            updateConversationInList(conversation)
        }

        isGenerating = false
        streamingContent = ""
    }

    /// Cancel ongoing generation
    func cancelGeneration() {
        inferenceService.cancelGeneration()
    }

    /// Update the streaming message in the current conversation
    private func updateStreamingMessage() {
        guard var conversation = currentConversation else { return }

        conversation.updateLastAssistantMessage(
            content: streamingContent,
            windowShifts: currentWindowShifts,
            historyTokens: currentHistoryTokens > 0 ? currentHistoryTokens : nil
        )

        currentConversation = conversation
    }

    /// Update a conversation in the list
    private func updateConversationInList(_ conversation: Conversation) {
        if let index = conversations.firstIndex(where: { $0.id == conversation.id }) {
            conversations[index] = conversation
        }
    }

    // MARK: - Model Management

    /// Check if model is loaded
    var isModelLoaded: Bool {
        inferenceService.isModelLoaded
    }

    /// Current model loading progress
    var modelLoadingProgress: ModelLoadingProgress? {
        inferenceService.loadingProgress
    }

    /// Load a model
    func loadModel(from path: URL) async throws {
        try await inferenceService.loadModel(from: path)
    }

    /// Unload the current model
    func unloadModel() async {
        await inferenceService.unloadModel()
    }

    // MARK: - Settings

    /// Get current temperature
    var temperature: Float {
        get { inferenceService.temperature }
        set { inferenceService.temperature = newValue }
    }

    /// Get current max tokens
    var maxTokens: Int {
        get { inferenceService.maxTokens }
        set { inferenceService.maxTokens = newValue }
    }

    /// Get current system prompt
    var systemPrompt: String {
        get { inferenceService.systemPrompt }
        set { inferenceService.systemPrompt = newValue }
    }

    /// Save settings
    func saveSettings() async {
        await StorageService.shared.saveTemperature(temperature)
        await StorageService.shared.saveMaxTokens(maxTokens)
        await StorageService.shared.saveSystemPrompt(systemPrompt)
    }
}
