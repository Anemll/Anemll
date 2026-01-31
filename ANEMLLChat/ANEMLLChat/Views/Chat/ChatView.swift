//
//  ChatView.swift
//  ANEMLLChat
//
//  Main chat interface
//

import SwiftUI

struct ChatView: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(ModelManagerViewModel.self) private var modelManager

    @State private var scrollProxy: ScrollViewProxy?
    @State private var isUserScrolling = false
    @State private var showingModelSheet = false

    var body: some View {
        VStack(spacing: 0) {
            // Messages
            messagesView

            // Model loading indicator
            if let progress = modelManager.loadingProgress, modelManager.isLoadingModel {
                ModelLoadingBar(progress: progress)
            }

            // Input bar
            InputBar()
                .environment(chatVM)
        }
        .navigationTitle(chatVM.currentConversation?.title ?? "Chat")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                modelStatusButton
            }
        }
        .sheet(isPresented: $showingModelSheet) {
            ModelListView()
                .environment(modelManager)
                .environment(chatVM)
        }
        .alert("Error", isPresented: .constant(chatVM.errorMessage != nil)) {
            Button("OK") {
                chatVM.errorMessage = nil
            }
        } message: {
            if let error = chatVM.errorMessage {
                Text(error)
            }
        }
    }

    // MARK: - Messages View

    private var messagesView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(visibleMessages) { message in
                        MessageBubble(message: message)
                            .id(message.id)
                    }

                    // Typing indicator
                    if chatVM.isGenerating && !chatVM.streamingContent.isEmpty {
                        typingIndicator
                    }
                }
                .padding()
            }
            .onAppear {
                scrollProxy = proxy
            }
            .onChange(of: chatVM.currentConversation?.messages.count) { _, _ in
                scrollToBottom()
            }
            .onChange(of: chatVM.streamingContent) { _, _ in
                if !isUserScrolling {
                    scrollToBottom()
                }
            }
        }
        .background(Color(platformBackground))
    }

    private var visibleMessages: [ChatMessage] {
        chatVM.currentConversation?.messages.filter { $0.role != .system } ?? []
    }

    private var typingIndicator: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle()
                    .fill(Color.secondary)
                    .frame(width: 6, height: 6)
                    .scaleEffect(chatVM.isGenerating ? 1.0 : 0.5)
                    .animation(
                        .easeInOut(duration: 0.5)
                        .repeatForever()
                        .delay(Double(i) * 0.2),
                        value: chatVM.isGenerating
                    )
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(platformSecondaryBackground), in: Capsule())
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func scrollToBottom() {
        guard let lastMessage = visibleMessages.last else { return }

        withAnimation(.easeOut(duration: 0.2)) {
            scrollProxy?.scrollTo(lastMessage.id, anchor: .bottom)
        }
    }

    // MARK: - Model Status

    private var modelStatusButton: some View {
        Button {
            showingModelSheet = true
        } label: {
            HStack(spacing: 6) {
                Circle()
                    .fill(modelManager.loadedModelId != nil ? Color.green : Color.orange)
                    .frame(width: 10, height: 10)

                if let modelId = modelManager.loadedModelId,
                   let model = modelManager.availableModels.first(where: { $0.id == modelId }) {
                    Text(model.name)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .lineLimit(1)
                } else {
                    Text("No Model")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Image(systemName: "chevron.down")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color(platformSecondaryBackground), in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Model Loading Bar

struct ModelLoadingBar: View {
    let progress: ModelLoadingProgress

    var body: some View {
        VStack(spacing: 4) {
            HStack {
                Text(progress.stage)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                Text("\(Int(progress.percentage * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            ProgressView(value: progress.percentage)
                .progressViewStyle(.linear)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(platformSecondaryBackground))
    }
}

// MARK: - Platform Colors

#if os(iOS)
private let platformBackground = UIColor.systemBackground
private let platformSecondaryBackground = UIColor.secondarySystemBackground
#else
private let platformBackground = NSColor.windowBackgroundColor
private let platformSecondaryBackground = NSColor.controlBackgroundColor
#endif

#Preview {
    NavigationStack {
        ChatView()
            .environment(ChatViewModel())
            .environment(ModelManagerViewModel())
    }
}
