// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// MessagesContainerView.swift

import SwiftUI

struct MessagesContainerView: View {
    @ObservedObject var chat: Chat
    @Binding var isTyping: Bool
    @Binding var showCopiedFeedback: Bool
    @Binding var isAtBottom: Bool
    @Binding var scrollingTimer: Timer?
    @Binding var scrollProxy: ScrollViewProxy?
    @Binding var contentHeight: CGFloat
    @Binding var scrollViewHeight: CGFloat
    @Binding var forceScrollTrigger: Bool

    // Track the last message count to detect new messages
    @State private var lastMessageCount: Int = 0
    // Debounce timer for scroll operations
    @State private var scrollDebounceTimer: Timer?
    // Track if user has manually scrolled up (to avoid auto-scroll interrupting reading)
    @State private var userScrolledUp: Bool = false
    // Track last known scroll position
    @State private var lastScrollOffset: CGFloat = 0

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                messageContent
                    .background(
                        GeometryReader { geo in
                            Color.clear
                                .preference(key: ScrollOffsetPreferenceKey.self, value: geo.frame(in: .named("scroll")).minY)
                        }
                    )
            }
            .coordinateSpace(name: "scroll")
            .onPreferenceChange(ScrollOffsetPreferenceKey.self) { offset in
                // Detect if user scrolled up manually - works even during typing
                let scrolledUp = offset > lastScrollOffset + 30 // 30pt threshold for more responsive detection
                if scrolledUp {
                    userScrolledUp = true
                }
                lastScrollOffset = offset
            }
            .frame(maxHeight: .infinity)
            .onAppear {
                scrollProxy = proxy
                lastMessageCount = chat.messages.count

                // Initial scroll to bottom
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                    smoothScrollToBottom(proxy: proxy, animated: false)
                }
            }
            .onDisappear {
                scrollDebounceTimer?.invalidate()
                scrollDebounceTimer = nil
            }
            .onChange(of: chat.messages.count) { oldCount, newCount in
                // New message added
                if newCount > oldCount {
                    // Only reset scroll state if it's a user message (user just sent something)
                    // Don't reset if AI is generating - that would override user's scroll position
                    if let lastMessage = chat.messages.last, lastMessage.isUser {
                        userScrolledUp = false
                    }
                    scheduleScroll(proxy: proxy, animated: true)
                }
            }
            .onChange(of: chat.messages.last?.text) { _, _ in
                // Content of last message changed (streaming)
                if isTyping && !userScrolledUp {
                    scheduleScroll(proxy: proxy, animated: false)
                }
            }
            .onChange(of: isTyping) { _, newValue in
                if newValue {
                    // Started typing - only scroll if user hasn't scrolled up
                    // Don't reset userScrolledUp here - respect user's reading position
                    if !userScrolledUp {
                        scheduleScroll(proxy: proxy, animated: true)
                    }
                } else {
                    // Stopped typing - only do final scroll if user is following along
                    if !userScrolledUp {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                            smoothScrollToBottom(proxy: proxy, animated: true)
                        }
                    }
                }
            }
            .onChange(of: forceScrollTrigger) { _, _ in
                userScrolledUp = false
                smoothScrollToBottom(proxy: proxy, animated: true)
            }
            .overlay(alignment: .bottomTrailing) {
                scrollToBottomButton(proxy: proxy)
            }
        }
    }

    // Debounced scroll - prevents multiple rapid scroll calls
    private func scheduleScroll(proxy: ScrollViewProxy, animated: Bool) {
        // Skip if user has scrolled up - respect their reading position
        guard !userScrolledUp else { return }

        scrollDebounceTimer?.invalidate()
        // Increased debounce to 0.2s to reduce UI load during fast token generation
        scrollDebounceTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: false) { _ in
            DispatchQueue.main.async {
                self.smoothScrollToBottom(proxy: proxy, animated: animated)
            }
        }
    }

    // Single smooth scroll implementation
    private func smoothScrollToBottom(proxy: ScrollViewProxy, animated: Bool) {
        guard !userScrolledUp else { return }

        let scrollAction = {
            proxy.scrollTo("bottom", anchor: .bottom)
            isAtBottom = true
        }

        if animated {
            withAnimation(.easeOut(duration: 0.25)) {
                scrollAction()
            }
        } else {
            scrollAction()
        }
    }
    
    private var messageContent: some View {
        LazyVStack(spacing: 4) {
            ForEach(chat.messages) { message in
                VStack(alignment: .leading, spacing: 4) {
                    // Message bubble
                    MessageContainerBubble(message: message, isGenerating: isTyping)

                    // Add tokens per second indicator for non-user messages
                    if !message.isUser && !isTyping, let tps = message.tokensPerSecond {
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
                .padding(.vertical, 4)
                .padding(message.isUser ? .leading : .horizontal, message.isUser ? 50 : 0)
                .onTapGesture(count: 2) {
                    copyMessage(message)
                }
                .id(message.id)
            }

            if isTyping {
                typingIndicator
            }

            // Bottom anchor for scrolling - increased height for better detection
            Color.clear
                .frame(height: 20)
                .id("bottom")
        }
        .padding(.horizontal)
    }

    private var typingIndicator: some View {
        HStack(spacing: 8) {
            ProgressView()
                .scaleEffect(0.8)
            Text("Generating...")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .id("typing")
    }

    private func scrollToBottomButton(proxy: ScrollViewProxy) -> some View {
        Group {
            if !isAtBottom || userScrolledUp {
                Button(action: {
                    userScrolledUp = false
                    smoothScrollToBottom(proxy: proxy, animated: true)
                }) {
                    Image(systemName: "arrow.down.circle.fill")
                        .font(.system(size: 32))
                        .foregroundColor(.blue)
                        .background(Circle().fill(Color(.systemBackground)))
                        .shadow(radius: 2)
                }
                .padding(.trailing, 16)
                .padding(.bottom, 16)
                .transition(.scale.combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.2), value: isAtBottom)
        .animation(.easeInOut(duration: 0.2), value: userScrolledUp)
    }

    private func copyMessage(_ message: Message) {
        UIPasteboard.general.string = message.text
        withAnimation {
            showCopiedFeedback = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            showCopiedFeedback = false
        }
    }
}

// Preference key for tracking scroll offset
private struct ScrollOffsetPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

// MessageBubble implementation for MessagesContainerView
private struct MessageContainerBubble: View {
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
                    .background(Color(.systemBackground))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                    )
                    .cornerRadius(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .id(message.id)
        .transition(.opacity)
    }
} 
