// Copyright (c) 2025 Anemll
// Licensed under the MIT License

// Chat.swift
import Foundation
import SwiftUI

// Change from struct to class for reference semantics
class Chat: Identifiable, Codable, ObservableObject {
    var id: String = UUID().uuidString
    var modelId: String
    @Published var messages: [Message]
    var createdAt: Date = Date()

    var title: String {
        "Chat \(createdAt.formatted(date: .numeric, time: .shortened))"
    }

    var lastMessagePreview: String {
        messages.last?.text ?? ""
    }
    
    init(id: String = UUID().uuidString, modelId: String, messages: [Message] = [], createdAt: Date = Date()) {
        self.id = id
        self.modelId = modelId
        self.messages = messages
        self.createdAt = createdAt
    }
    
    // Add a method to add a message
    func addMessage(_ message: Message) {
        if Thread.isMainThread {
            messages.append(message)
            print("DEBUG - Added message to chat: \(message.isUser ? "User" : "AI") message, ID: \(message.id)")
            objectWillChange.send()
        } else {
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.messages.append(message)
                print("DEBUG - Added message to chat: \(message.isUser ? "User" : "AI") message, ID: \(message.id)")
                self.objectWillChange.send()
            }
        }
    }
    
    // Method to update message text at a specific index
    func updateMessage(at index: Int, with text: String) -> Bool {
        // Validate the index
        guard index >= 0 && index < messages.count else {
            print("DEBUG - Invalid index \(index) for message update")
            return false
        }
        
        // If the text is identical, avoid triggering unnecessary updates
        if messages[index].text == text {
            return true
        }
        
        if Thread.isMainThread {
            // Direct update without animation to reduce overhead
            messages[index].updateText(text)
            // Only trigger a change notification for the chat, not for both message and chat
            objectWillChange.send()
            return true
        } else {
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.messages[index].updateText(text)
                // Only trigger a change notification for the chat
                self.objectWillChange.send()
            }
            return true
        }
    }
    
    // Add a method to get the last AI message index
    func lastAIMessageIndex() -> Int? {
        return messages.lastIndex(where: { !$0.isUser })
    }
    
    // Required for Codable when using @Published
    enum CodingKeys: String, CodingKey {
        case id, modelId, messages, createdAt
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        modelId = try container.decode(String.self, forKey: .modelId)
        messages = try container.decode([Message].self, forKey: .messages)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(modelId, forKey: .modelId)
        try container.encode(messages, forKey: .messages)
        try container.encode(createdAt, forKey: .createdAt)
    }
}

extension Chat: Hashable {
    static func == (lhs: Chat, rhs: Chat) -> Bool {
        return lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
