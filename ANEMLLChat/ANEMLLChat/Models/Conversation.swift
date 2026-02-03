//
//  Conversation.swift
//  ANEMLLChat
//
//  Conversation container for chat history
//

import Foundation

/// A chat conversation containing messages
struct Conversation: Identifiable, Codable, Sendable {
    let id: UUID
    var title: String
    var messages: [ChatMessage]
    var modelId: String?
    let createdAt: Date
    var updatedAt: Date

    init(
        id: UUID = UUID(),
        title: String = "New Chat",
        messages: [ChatMessage] = [],
        modelId: String? = nil,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.modelId = modelId
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    /// Add a message to the conversation
    mutating func addMessage(_ message: ChatMessage) {
        messages.append(message)
        updatedAt = Date()

        // Auto-generate title from first user message
        if title == "New Chat", message.role == .user {
            title = String(message.content.prefix(40))
            if message.content.count > 40 {
                title += "..."
            }
        }
    }

    /// Update the last assistant message (for streaming)
    mutating func updateLastAssistantMessage(
        content: String? = nil,
        tokensPerSecond: Double? = nil,
        tokenCount: Int? = nil,
        windowShifts: Int? = nil,
        prefillTime: TimeInterval? = nil,
        prefillTokens: Int? = nil,
        historyTokens: Int? = nil,
        isComplete: Bool? = nil,
        wasCancelled: Bool? = nil,
        stopReason: String? = nil
    ) {
        guard let index = messages.lastIndex(where: { $0.role == .assistant }) else { return }

        if let content = content {
            messages[index].content = content
        }
        if let tps = tokensPerSecond {
            messages[index].tokensPerSecond = tps
        }
        if let count = tokenCount {
            messages[index].tokenCount = count
        }
        if let shifts = windowShifts {
            messages[index].windowShifts = shifts
        }
        if let time = prefillTime {
            messages[index].prefillTime = time
        }
        if let tokens = prefillTokens {
            messages[index].prefillTokens = tokens
        }
        if let history = historyTokens {
            messages[index].historyTokens = history
        }
        if let complete = isComplete {
            messages[index].isComplete = complete
        }
        if let cancelled = wasCancelled {
            messages[index].wasCancelled = cancelled
        }
        if let reason = stopReason {
            messages[index].stopReason = reason
        }

        updatedAt = Date()
    }

    /// Get the last assistant message index
    var lastAssistantMessageIndex: Int? {
        messages.lastIndex(where: { $0.role == .assistant })
    }

    /// Check if conversation is empty (no user messages)
    var isEmpty: Bool {
        !messages.contains(where: { $0.role == .user })
    }

    /// Get message count excluding system messages
    var visibleMessageCount: Int {
        messages.filter { $0.role != .system }.count
    }
}

// MARK: - Formatting Helpers

extension Conversation {
    /// Format the last update time
    var formattedDate: String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: updatedAt, relativeTo: Date())
    }

    /// Preview of the last message
    var lastMessagePreview: String? {
        guard let lastMessage = messages.last(where: { $0.role != .system && !$0.content.isEmpty }) else {
            return nil
        }
        let preview = lastMessage.content.prefix(60)
        return preview.count < lastMessage.content.count ? "\(preview)..." : String(preview)
    }
}
