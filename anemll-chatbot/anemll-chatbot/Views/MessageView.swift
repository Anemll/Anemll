// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// MessageView.swift

import SwiftUI

struct MessageView: View {
    @ObservedObject var message: Message
    var isGenerating: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            // Message bubble
            MessageBubble(message: message, isGenerating: isGenerating)
            
            // Add tokens per second indicator for non-user messages
            if !message.isUser && !isGenerating, let tps = message.tokensPerSecond {
                Text(String(format: "%.1f tokens/sec", tps))
                    .font(.system(size: 10))
                    .foregroundColor(.blue)
                    .padding(.leading, 8)
            }
            
            // Add a window shift indicator for long generations
            if !message.isUser && message.windowShifts > 0 {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .foregroundColor(.blue)
                        .font(.system(size: 12))
                    
                    Text("Long generation (\(message.windowShifts) shifts)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 4)
                .padding(.leading, 12)
                .transition(.opacity)
            }
        }
    }
}

// Add the MessageBubble that is referenced in this view
struct MessageBubble: View {
    @ObservedObject var message: Message
    var isGenerating: Bool
    
    var body: some View {
        HStack {
            if message.isSystemMessage {
                // System message styling
                HStack {
                    Image(systemName: "info.circle.fill")
                        .foregroundColor(.blue)
                    Text(message.text)
                        .foregroundColor(.primary)
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(12)
                .frame(maxWidth: .infinity, alignment: .center)
            } else if message.isUser {
                Spacer()
                Text(message.text)
                    .padding()
                    .background(Color.blue.opacity(0.2))
                    .cornerRadius(12)
            } else {
                Text(message.text.isEmpty ? "..." : message.text)
                    .padding()
                    .background(Color.white)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.blue.opacity(0.3), lineWidth: 2)
                    )
                    .cornerRadius(12)
                    .shadow(color: Color.gray.opacity(0.1), radius: 2, x: 0, y: 1)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .id(message.id)
        .transition(.opacity)
    }
}

#Preview {
    VStack(spacing: 12) {
        MessageView(
            message: Message(text: "Hello, how can I help you?", isUser: false),
            isGenerating: false
        )
        
        MessageView(
            message: {
                let msg = Message(text: "This is a long generation with window shifts", isUser: false)
                msg.windowShifts = 2
                return msg
            }(),
            isGenerating: false
        )
        
        MessageView(
            message: Message(text: "What can you tell me about window management?", isUser: true),
            isGenerating: false
        )
    }
    .padding()
} 
