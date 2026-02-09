//
//  MessageBubble.swift
//  ANEMLLChat
//
//  Individual message display
//

import SwiftUI
#if os(macOS)
import AppKit
#else
import UIKit
#endif

struct MessageBubble: View, Equatable {
    let message: ChatMessage
    let allowSelection: Bool
    let enableMarkup: Bool

    @State private var isHovering = false
    @State private var showCopyButton = false

    private var isUser: Bool {
        message.role == .user
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Message content (overlay is inside messageContent now)
            messageContent
            #if os(iOS)
            .onTapGesture {
                withAnimation(.easeInOut(duration: 0.2)) {
                    showCopyButton.toggle()
                }
            }
            #endif

            // Stats (for assistant messages)
            if !isUser && message.isComplete {
                statsView
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        #if os(macOS)
        .contentShape(Rectangle())  // Make entire area hoverable
        .onHover { hovering in
            // No animation on hover change to prevent flicker
            isHovering = hovering
        }
        #endif
    }

    // MARK: - Copy Button

    @ViewBuilder
    private var copyButton: some View {
        Button {
            copyToClipboard()
            #if os(iOS)
            withAnimation {
                showCopyButton = false
            }
            #endif
        } label: {
            Image(systemName: "doc.on.doc")
                .font(.caption)
                .padding(5)
                .modifier(CopyButtonGlassModifier())
        }
        .buttonStyle(.plain)
    }

    private func copyToClipboard() {
        copyText(message.content)
    }

    /// Copy text excluding <think>...</think> blocks
    private func copyWithoutThinking() {
        let cleaned = stripThinkBlocks(from: message.content)
        copyText(cleaned)
    }

    private func copyText(_ text: String) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #else
        UIPasteboard.general.string = text
        #endif
    }

    /// Remove <think>...</think> blocks from content
    private func stripThinkBlocks(from content: String) -> String {
        var result = content

        // Remove complete <think>...</think> blocks
        let completePattern = #"<think>[\s\S]*?</think>"#
        if let regex = try? NSRegularExpression(pattern: completePattern, options: []) {
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "")
        }

        // Remove incomplete <think>... (streaming case)
        if let openRange = result.range(of: "<think>") {
            result = String(result[..<openRange.lowerBound])
        }

        // Clean up extra whitespace
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Message Content

    private var messageContent: some View {
        HStack(alignment: .top, spacing: 12) {
            RoundedRectangle(cornerRadius: 2, style: .continuous)
                .fill(isUser ? Color.accentColor.opacity(0.9) : llmAccent)
                .frame(width: 3)

            // Text content with copy button overlay
            // Use HStack to keep button close to text end
            HStack(alignment: .top, spacing: 4) {
                VStack(alignment: .leading, spacing: 8) {
                    if message.content.isEmpty && !message.isComplete {
                        // Loading state
                        ProgressView()
                            .controlSize(.small)
                    } else if isUser {
                        // User messages - simple text
                        Text(message.content)
                            .selectable(allowSelection)
                            .lineSpacing(3)
                    } else {
                        // Assistant messages
                        if enableMarkup {
                            MarkdownView(content: message.content, isUserMessage: false, allowSelection: allowSelection, isMessageComplete: message.isComplete)
                        } else {
                            Text(message.content)
                                .selectable(allowSelection)
                                .lineSpacing(3)
                        }
                    }
                }

                // Copy button inline (appears on hover, no text reflow since it's always in layout)
                if !message.content.isEmpty {
                    copyButton
                        .opacity(shouldShowCopyButton ? 1 : 0)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 6)
        .foregroundStyle(.primary)
    }

    private var shouldShowCopyButton: Bool {
        #if os(macOS)
        return isHovering
        #else
        return showCopyButton
        #endif
    }

    static func == (lhs: MessageBubble, rhs: MessageBubble) -> Bool {
        lhs.message == rhs.message && lhs.allowSelection == rhs.allowSelection && lhs.enableMarkup == rhs.enableMarkup
    }
}

private let llmAccent = Color(red: 1.0, green: 0.62, blue: 0.2)

// MARK: - Stats View

extension MessageBubble {
    @ViewBuilder
    fileprivate var statsView: some View {
        // Always show key stats: copy button, tok/s, prefill speed, context tokens
        HStack(spacing: 8) {
            // Copy button (always visible in stats row) - excludes thinking
            Button {
                copyWithoutThinking()
            } label: {
                Image(systemName: "doc.on.doc")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Copy message (excludes thinking)")

            // Generation speed
            if let tps = message.tokensPerSecond {
                HStack(spacing: 2) {
                    Image(systemName: "gauge.medium")
                        .font(.caption2)
                    Text(String(format: "%.1f tok/s", tps))
                        .font(.caption2)
                }
                .foregroundStyle(.secondary)
            }

            // Prefill speed (TTFT)
            if let prefillTime = message.prefillTime, let prefillTokens = message.prefillTokens, prefillTime > 0 {
                let prefillSpeed = Double(prefillTokens) / prefillTime
                HStack(spacing: 2) {
                    Image(systemName: "arrow.right.circle")
                        .font(.caption2)
                    Text(String(format: "%.0f t/s", prefillSpeed))
                        .font(.caption2)
                }
                .foregroundStyle(.cyan)
            }

            // History token count (matches CLI)
            if let ctx = message.historyTokens {
                HStack(spacing: 2) {
                    Image(systemName: "text.alignleft")
                        .font(.caption2)
                    Text("\(ctx) ctx")
                        .font(.caption2)
                }
                .foregroundStyle(.green)
            }
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

// MARK: - Glass Effect Modifier (macOS 26+)

private struct CopyButtonGlassModifier: ViewModifier {
    func body(content: Content) -> some View {
        #if os(macOS)
        if #available(macOS 26.0, *) {
            content
                .glassEffect(.regular.interactive())
                .clipShape(Circle())
        } else {
            content
                .background(.ultraThinMaterial, in: Circle())
        }
        #else
        content
            .background(.ultraThinMaterial, in: Circle())
        #endif
    }
}

// MARK: - Preview

#Preview {
    VStack(spacing: 16) {
        MessageBubble(message: .user("Hello! How are you today?"), allowSelection: true, enableMarkup: true)

        MessageBubble(message: ChatMessage(
            role: .assistant,
            content: "I'm doing great, thank you for asking! How can I help you today?",
            tokensPerSecond: 24.5,
            tokenCount: 15,
            prefillTime: 0.05,
            prefillTokens: 10,
            historyTokens: 25,
            isComplete: true
        ), allowSelection: true, enableMarkup: true)

        MessageBubble(message: ChatMessage(
            role: .assistant,
            content: "This is a longer response with **markdown** support and `code blocks`.",
            tokensPerSecond: 18.2,
            tokenCount: 50,
            windowShifts: 2,
            prefillTime: 0.15,
            prefillTokens: 128,
            historyTokens: 178,
            isComplete: true
        ), allowSelection: true, enableMarkup: true)

        MessageBubble(message: ChatMessage(
            role: .assistant,
            content: "",
            isComplete: false
        ), allowSelection: true, enableMarkup: true)
    }
    .padding()
}
