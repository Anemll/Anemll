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
    @State private var showScrollToBottom = false

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
        // Error toast (non-intrusive)
        .errorToast(Binding(
            get: { chatVM.errorMessage },
            set: { chatVM.errorMessage = $0 }
        ))
    }

    // MARK: - Messages View

    private var messagesView: some View {
        GeometryReader { outerGeometry in
            ZStack(alignment: .bottom) {
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

                            // Bottom anchor for scroll detection
                            GeometryReader { bottomGeometry in
                                Color.clear
                                    .preference(
                                        key: BottomVisiblePreferenceKey.self,
                                        value: bottomGeometry.frame(in: .global).minY < outerGeometry.frame(in: .global).maxY + 50
                                    )
                            }
                            .frame(height: 1)
                            .id("bottom")
                        }
                        .padding(.horizontal)
                        .padding(.top)
                        .padding(.bottom, 20) // Extra bottom padding to avoid overlap with input bar
                    }
                    .onPreferenceChange(BottomVisiblePreferenceKey.self) { isBottomVisible in
                        // Show button when bottom anchor is NOT visible (scrolled up)
                        showScrollToBottom = !isBottomVisible && !visibleMessages.isEmpty
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

                // Scroll to bottom button
                if showScrollToBottom {
                    Button {
                        scrollToBottom()
                    } label: {
                        Image(systemName: "chevron.down")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.primary)
                            .frame(width: 36, height: 36)
                            .background(Color(platformSecondaryBackground))
                            .clipShape(Circle())
                            .shadow(color: .black.opacity(0.15), radius: 4, y: 2)
                    }
                    .padding(.bottom, 12)
                    .transition(.scale.combined(with: .opacity))
                }
            }
        }
        .background(Color(platformBackground))
        .animation(.easeInOut(duration: 0.2), value: showScrollToBottom)
    }

    // Preference key for tracking if bottom is visible
    private struct BottomVisiblePreferenceKey: PreferenceKey {
        static var defaultValue: Bool = true
        static func reduce(value: inout Bool, nextValue: () -> Bool) {
            value = nextValue()
        }
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
