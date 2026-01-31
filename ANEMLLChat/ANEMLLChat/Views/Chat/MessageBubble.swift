//
//  MessageBubble.swift
//  ANEMLLChat
//
//  Individual message display
//

import SwiftUI

struct MessageBubble: View {
    let message: ChatMessage

    @State private var showStats = false

    private var isUser: Bool {
        message.role == .user
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if isUser {
                Spacer(minLength: 60)
            }

            VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
                // Message content
                messageContent

                // Stats (for assistant messages)
                if !isUser && message.isComplete {
                    statsView
                }
            }

            if !isUser {
                Spacer(minLength: 60)
            }
        }
    }

    // MARK: - Message Content

    private var messageContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            if message.content.isEmpty && !message.isComplete {
                // Loading state
                ProgressView()
                    .controlSize(.small)
            } else {
                // Text content with markdown support
                Text(formattedContent)
                    .textSelection(.enabled)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(bubbleBackground)
        .foregroundStyle(isUser ? .white : .primary)
    }

    private var formattedContent: AttributedString {
        // Try to parse as markdown
        if let attributed = try? AttributedString(markdown: message.content) {
            return attributed
        }
        return AttributedString(message.content)
    }

    private var bubbleBackground: some View {
        RoundedRectangle(cornerRadius: 18, style: .continuous)
            .fill(isUser ? Color.accentColor : Color(platformSecondaryBackground))
    }
}

// MARK: - Platform Colors

#if os(iOS)
private let platformSecondaryBackground = UIColor.secondarySystemBackground
#else
private let platformSecondaryBackground = NSColor.controlBackgroundColor
#endif

// MARK: - Stats View

extension MessageBubble {
    @ViewBuilder
    fileprivate var statsView: some View {
        if let stats = message.performanceStats {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    showStats.toggle()
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "gauge.medium")
                        .font(.caption2)

                    if showStats {
                        Text(stats)
                            .font(.caption2)
                    } else if let tps = message.tokensPerSecond {
                        Text(String(format: "%.1f tok/s", tps))
                            .font(.caption2)
                    }
                }
                .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }

        // Window shifts indicator
        if let shifts = message.windowShifts, shifts > 0 {
            HStack(spacing: 4) {
                Image(systemName: "arrow.left.arrow.right")
                    .font(.caption2)
                Text("\(shifts) context shifts")
                    .font(.caption2)
            }
            .foregroundStyle(.orange)
        }

        // Cancelled indicator
        if message.wasCancelled {
            HStack(spacing: 4) {
                Image(systemName: "stop.circle")
                    .font(.caption2)
                Text("Cancelled")
                    .font(.caption2)
            }
            .foregroundStyle(.orange)
        }
    }
}

// MARK: - Preview

#Preview {
    VStack(spacing: 16) {
        MessageBubble(message: .user("Hello! How are you today?"))

        MessageBubble(message: ChatMessage(
            role: .assistant,
            content: "I'm doing great, thank you for asking! How can I help you today?",
            tokensPerSecond: 24.5,
            tokenCount: 15,
            isComplete: true
        ))

        MessageBubble(message: ChatMessage(
            role: .assistant,
            content: "This is a longer response with **markdown** support and `code blocks`.",
            tokensPerSecond: 18.2,
            tokenCount: 50,
            windowShifts: 2,
            isComplete: true
        ))

        MessageBubble(message: ChatMessage(
            role: .assistant,
            content: "",
            isComplete: false
        ))
    }
    .padding()
}
