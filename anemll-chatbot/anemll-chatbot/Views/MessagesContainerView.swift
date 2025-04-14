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
    
    // Add state to track the latest message ID for scrolling
    @State private var lastMessageId: UUID?
    // Track if we've already scrolled to the latest message
    @State private var hasScrolledToLatest: Bool = false
    // Track the last message count to detect actual changes
    @State private var lastMessageCount: Int = 0
    // Track if we're processing a scroll operation
    @State private var isScrolling: Bool = false
    // Track the last content of the latest message to detect content changes
    @State private var lastMessageContent: String = ""
    
    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                messageContent
                    .onChange(of: chat.messages) { _, newMessages in
                        print("DEBUG: MessagesContainerView detected message changes, count: \(newMessages.count)")
                        // Only update lastMessageId if there's a new message
                        if newMessages.count != lastMessageCount {
                            lastMessageCount = newMessages.count
                            if let lastMessage = newMessages.last {
                                lastMessageId = lastMessage.id
                                lastMessageContent = lastMessage.text
                                print("DEBUG: New last message ID: \(lastMessageId?.uuidString ?? "nil"), text: \(lastMessageContent)")
                                
                                // Only scroll automatically if we're already at the bottom
                                // or if we're about to start typing
                                if isAtBottom || isTyping {
                                    // Give time for the new message view to be created
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                                        scrollToBottom(animated: true)
                                    }
                                }
                            }
                        } else if let lastMessage = newMessages.last, lastMessage.text != lastMessageContent {
                            // Detect if the content of the last message changed (during typing)
                            lastMessageContent = lastMessage.text
                            
                            // If we're typing, scroll to the bottom to follow the new content
                            if isTyping {
                                scrollToBottom(animated: false)
                            }
                        }
                    }
            }
            .frame(maxHeight: .infinity)
            .onAppear { 
                print("DEBUG: ScrollView appeared with \(chat.messages.count) messages")
                scrollProxy = proxy
                lastMessageCount = chat.messages.count
                
                // Save the content of the last message if there is one
                if let lastMessage = chat.messages.last {
                    lastMessageContent = lastMessage.text
                    lastMessageId = lastMessage.id
                    print("DEBUG: Initial lastMessageId: \(lastMessageId?.uuidString ?? "nil"), text: \(lastMessageContent)")
                }
                
                // Initial scroll to the bottom when the view appears
                // Use a small delay to ensure the view is fully rendered
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                    if let lastMessage = chat.messages.last {
                        lastMessageId = lastMessage.id
                        print("DEBUG: Attempting to scroll to last message: \(lastMessageId?.uuidString ?? "nil")")
                        scrollToBottom(animated: false)
                    }
                }
            }
            .onDisappear {
                print("DEBUG: ScrollView disappeared")
                stopScrollingTimer()
            }
            .onChange(of: isTyping) { _, newValue in
                print("DEBUG: isTyping changed to \(newValue)")
                if newValue {
                    // Start typing - wait for the typing indicator to be added to the view
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        // First scroll to the typing indicator
                        scrollToTypingIndicator()
                        
                        // Then start the continuous scrolling timer with a short interval
                        startScrollingTimer()
                    }
                } else {
                    // Stop typing - cancel any auto-scrolling
                    stopScrollingTimer()
                    
                    // Single, final scroll with a longer delay to ensure message is rendered
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                        if let lastMessage = chat.messages.last {
                            withAnimation(.easeInOut(duration: 0.3)) {
                                scrollProxy?.scrollTo(lastMessage.id, anchor: .bottom)
                            }
                        }
                    }
                }
            }
            .overlay(alignment: .bottomTrailing) {
                scrollToBottomButton
            }
            
            // Add a direct observer for the last message content to trigger scrolling during typing
            .onChange(of: lastMessageContent) { _, newContent in
                print("DEBUG: Last message content changed to: \(newContent.prefix(20))...")
                if isTyping {
                    scrollToBottom(animated: false)
                }
            }
            
            // Add observer for the force scroll trigger
            .onChange(of: forceScrollTrigger) { _, _ in
                print("DEBUG: forceScrollTrigger changed")
                // A change in forceScrollTrigger means we need to scroll immediately
                scrollToBottom(animated: true)
            }
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
                .id(message.id) // Use consistent ID for scrolling
                .onChange(of: message.text) { _, newText in
                    // Detect when a message's text changes (during AI response generation)
                    if !message.isUser && isTyping {
                        // For non-user messages during typing, scroll to bottom
                        scrollToBottom(animated: false)
                    }
                }
            }
            
            if isTyping {
                typingIndicator
            }
            
            // Add an empty spacer view with ID "bottom" for scrolling to bottom
            Color.clear
                .frame(height: 1)
                .id("bottom")
        }
        .padding(.horizontal)
    }
    
    private var typingIndicator: some View {
        HStack {
            ProgressView()
            //Text("Generating...")
            //Text("")
        }
        .padding()
        .id("typing")
    }
    
    private var scrollToBottomButton: some View {
        Group {
            if !isAtBottom {
                Button(action: { scrollToBottom(animated: true) }) {
                    Image(systemName: "arrow.down.circle.fill")
                        .font(.system(size: 30))
                        .foregroundColor(.blue)
                }
                .padding(.trailing, 10)
                .padding(.bottom, 10)
            }
        }
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
    
    private func scrollToBottom(animated: Bool = true) {
        guard let proxy = scrollProxy, !isScrolling else { 
            print("DEBUG: scrollToBottom - No proxy available or already scrolling")
            return
        }
        
        // Set scrolling flag to prevent multiple scrolls
        isScrolling = true
        print("DEBUG: scrollToBottom - Starting scroll operation, animated: \(animated)")
        
        let scrollAction = {
            if isTyping {
                // If typing, scroll to typing indicator
                print("DEBUG: scrollToBottom - Scrolling to typing indicator")
                proxy.scrollTo("typing", anchor: .bottom)
            } else if let lastMessageId = lastMessageId {
                print("DEBUG: scrollToBottom - Scrolling to last message ID: \(lastMessageId)")
                proxy.scrollTo(lastMessageId, anchor: .bottom)
            } else if let lastMessage = chat.messages.last {
                print("DEBUG: scrollToBottom - Scrolling to last message: \(lastMessage.id)")
                proxy.scrollTo(lastMessage.id, anchor: .bottom)
            } else {
                print("DEBUG: scrollToBottom - Scrolling to bottom marker")
                proxy.scrollTo("bottom", anchor: .bottom)
            }
            
            // Mark that we're at the bottom
            isAtBottom = true
        }
        
        if animated {
            withAnimation(.easeInOut(duration: 0.3)) {
                scrollAction()
            }
        } else {
            scrollAction()
        }
        
        // Reset scrolling flag after a short delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            isScrolling = false
            print("DEBUG: scrollToBottom - Scroll operation completed")
        }
    }
    
    private func scrollToTypingIndicator() {
        guard let proxy = scrollProxy, !isScrolling else { return }
        
        // Set scrolling flag to prevent multiple scrolls
        isScrolling = true
        
        withAnimation(.easeInOut(duration: 0.3)) {
            proxy.scrollTo("typing", anchor: .bottom)
            
            // Mark that we're at the bottom
            isAtBottom = true
        }
        
        // Reset scrolling flag after a delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            isScrolling = false
        }
    }
    
    private func startScrollingTimer() {
        // Only start the timer if we're typing and don't already have one
        if isTyping && scrollingTimer == nil {
            // Use a shorter interval for more responsive scrolling
            scrollingTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [self] _ in
                guard self.isTyping else { 
                    self.stopScrollingTimer()
                    return
                }
                
                // Only scroll if we're not already processing a scroll
                if !self.isScrolling {
                    self.scrollToBottom(animated: false)
                }
            }
        }
    }
    
    private func stopScrollingTimer() {
        scrollingTimer?.invalidate()
        scrollingTimer = nil
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
