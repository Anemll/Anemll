// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// Message.swift

import Foundation
import SwiftUI

// Change from struct to class for reference semantics
class Message: Identifiable, Codable, ObservableObject, Equatable {
    let id: UUID
    @Published var text: String
    var isUser: Bool
    var timestamp: Date
    var tokensPerSecond: Double?
    @Published var windowShifts: Int = 0 // Track window shifts during generation
    
    init(text: String, isUser: Bool) {
        self.id = UUID()
        self.text = text
        self.isUser = isUser
        self.timestamp = Date()
        self.tokensPerSecond = nil
        self.windowShifts = 0
    }
    
    // Add a method to update the text
    func updateText(_ newText: String) {
        // Make sure UI updates happen on the main thread
        if Thread.isMainThread {
            self.text = newText
            self.objectWillChange.send()
        } else {
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.text = newText
                self.objectWillChange.send()
            }
        }
    }
    
    // Implement Equatable
    static func == (lhs: Message, rhs: Message) -> Bool {
        let isIdEqual = lhs.id == rhs.id
        let isTextEqual = lhs.text == rhs.text
        let isUserEqual = lhs.isUser == rhs.isUser
        return isIdEqual && isTextEqual && isUserEqual
    }
    
    // Required for Codable when using @Published
    enum CodingKeys: String, CodingKey {
        case id, text, isUser, timestamp, tokensPerSecond, windowShifts
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        text = try container.decode(String.self, forKey: .text)
        isUser = try container.decode(Bool.self, forKey: .isUser)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        tokensPerSecond = try container.decodeIfPresent(Double.self, forKey: .tokensPerSecond)
        windowShifts = try container.decodeIfPresent(Int.self, forKey: .windowShifts) ?? 0
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(text, forKey: .text)
        try container.encode(isUser, forKey: .isUser)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encodeIfPresent(tokensPerSecond, forKey: .tokensPerSecond)
        try container.encode(windowShifts, forKey: .windowShifts)
    }
}
