//
//  ContentView.swift
//  ANEMLLChat
//
//  Root view with navigation
//

import SwiftUI

struct ContentView: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(ModelManagerViewModel.self) private var modelManager

    // Start with sidebar collapsed (detailOnly) on macOS
    @State private var columnVisibility: NavigationSplitViewVisibility = .detailOnly
    @State private var showingModelSheet = false
    @State private var showingSettings = false
    @State private var showingConversationSheet = false
    @State private var hasCheckedInitialState = false
    @State private var showingClearAllAlert = false

    var body: some View {
        #if os(iOS)
        iosRoot
        #else
        macRoot
        #endif
    }

    #if os(iOS)
    private var iosRoot: some View {
        NavigationStack {
            iosDetail
        }
        .sheet(isPresented: $showingConversationSheet) {
            ConversationListSheet {
                showingConversationSheet = false
            }
            .environment(chatVM)
        }
        .sheet(isPresented: $showingModelSheet) {
            ModelListView()
                .environment(modelManager)
                .environment(chatVM)
        }
        .sheet(isPresented: $showingSettings) {
            NavigationStack {
                SettingsView()
                    .environment(chatVM)
            }
        }
        // Auto-show model list on fresh start when no models are downloaded
        .task {
            guard !hasCheckedInitialState else { return }
            hasCheckedInitialState = true

            // Wait briefly for model list to load
            try? await Task.sleep(for: .milliseconds(500))

            // If no models are downloaded, show model list automatically
            if modelManager.downloadedModels.isEmpty && !modelManager.isLoadingModel {
                showingModelSheet = true
            }
        }
    }
    #else
    private var macRoot: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            sidebar
        } detail: {
            detail
        }
        .sheet(isPresented: $showingModelSheet) {
            ModelListView()
                .environment(modelManager)
                .environment(chatVM)
        }
        // Auto-show model list on fresh start when no models are downloaded
        .task {
            guard !hasCheckedInitialState else { return }
            hasCheckedInitialState = true

            // Wait briefly for model list to load
            try? await Task.sleep(for: .milliseconds(500))

            // If no models are downloaded, show model list automatically
            if modelManager.downloadedModels.isEmpty && !modelManager.isLoadingModel {
                showingModelSheet = true
            }
        }
    }
    #endif

    // MARK: - Sidebar

    private var sidebar: some View {
        @Bindable var vm = chatVM

        return List(selection: Binding(
            get: { chatVM.currentConversation?.id },
            set: { id in
                if let id, let conv = chatVM.conversations.first(where: { $0.id == id }) {
                    chatVM.selectConversation(conv)
                }
            }
        )) {
            Section {
                ForEach(chatVM.conversations) { conversation in
                    NavigationLink(value: conversation.id) {
                        ConversationRow(conversation: conversation)
                    }
                    .contextMenu {
                        Button(role: .destructive) {
                            chatVM.deleteConversation(conversation)
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                }
                .onDelete { indexSet in
                    chatVM.deleteConversation(at: indexSet)
                }
            } header: {
                Text("Conversations")
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("ANEMLL Chat")
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    chatVM.newConversation()
                } label: {
                    Label("New Chat", systemImage: "plus")
                }

                Button {
                    showingModelSheet = true
                } label: {
                    Label("Models", systemImage: "cpu")
                }
                .badge(modelManager.loadedModelId == nil ? "!" : nil)

                #if os(iOS)
                Button {
                    showingSettings = true
                } label: {
                    Label("Settings", systemImage: "gear")
                }
                #endif
            }

            #if os(macOS)
            ToolbarItem(placement: .destructiveAction) {
                Button(role: .destructive) {
                    showingClearAllAlert = true
                } label: {
                    Label("Clear All", systemImage: "trash")
                }
                .disabled(chatVM.conversations.isEmpty)
            }
            #endif
        }
        .alert("Clear All Conversations", isPresented: $showingClearAllAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Clear All", role: .destructive) {
                chatVM.clearAllConversations()
            }
        } message: {
            Text("This will delete all conversations. This action cannot be undone.")
        }
    }

    // MARK: - Detail

    @ViewBuilder
    private var detail: some View {
        VStack(spacing: 0) {
            // Top toolbar
            detailToolbar

            // Content
            if chatVM.currentConversation != nil {
                ChatView()
                    .environment(chatVM)
                    .environment(modelManager)
            } else {
                emptyState
            }
        }
    }

    #if os(iOS)
    private var iosDetail: some View {
        ZStack {
            ChatView()
                .environment(chatVM)
                .environment(modelManager)
        }
        .overlay(alignment: .top) {
            iosOverlayControls
        }
    }

    private var iosOverlayControls: some View {
        HStack {
            Spacer()
            HStack(spacing: 10) {
                Button {
                    chatVM.newConversation()
                } label: {
                    Image(systemName: "plus")
                }

                Button {
                    showingConversationSheet = true
                } label: {
                    Image(systemName: "list.bullet")
                }

                Button {
                    showingModelSheet = true
                } label: {
                    ZStack(alignment: .bottomTrailing) {
                        Image(systemName: "cpu")
                        Circle()
                            .fill(modelStatusColor)
                            .frame(width: 6, height: 6)
                            .offset(x: 3, y: 3)
                    }
                }

                Button {
                    showingSettings = true
                } label: {
                    Image(systemName: "gearshape")
                }
            }
            .font(.system(size: 14, weight: .semibold))
            .foregroundStyle(.primary)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(
                Capsule()
                    .stroke(Color.white.opacity(0.12), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.25), radius: 12, y: 6)
        }
        .padding(.top, 8)
        .padding(.horizontal, 12)
    }

    private var modelStatusColor: Color {
        if modelManager.isLoadingModel {
            return .blue
        }
        if modelManager.loadedModelId != nil {
            return .green
        }
        return .orange
    }
    #endif

    private var detailToolbar: some View {
        HStack(spacing: 8) {
            // New Chat button - icon only on iPhone for compactness
            Button {
                chatVM.newConversation()
            } label: {
                #if os(iOS)
                Image(systemName: "plus.bubble")
                    .imageScale(.large)
                #else
                Label("New Chat", systemImage: "plus.bubble")
                #endif
            }
            #if os(macOS)
            .buttonStyle(.accessoryBar)
            #else
            .buttonStyle(.bordered)
            #endif
            .controlSize(.small)

            // Download progress indicator (when downloading in background)
            if modelManager.downloadingModelId != nil {
                DownloadProgressPill(modelManager: modelManager) {
                    showingModelSheet = true
                }
            }

            Spacer()

            // Models button - compact pill style (combines model name + loading progress)
            Button {
                showingModelSheet = true
            } label: {
                HStack(spacing: 4) {
                    // Show loading model if loading, otherwise loaded model
                    if modelManager.isLoadingModel, let loadingId = modelManager.loadingModelId,
                       let model = modelManager.availableModels.first(where: { $0.id == loadingId }) {
                        // Loading state - show animated indicator + model name + progress
                        ProgressView()
                            .controlSize(.mini)
                        Text(model.name)
                            .font(.caption)
                            .lineLimit(1)
                        if let progress = modelManager.loadingProgress {
                            Text("\(Int(progress.percentage * 100))%")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    } else if let modelId = modelManager.loadedModelId,
                       let model = modelManager.availableModels.first(where: { $0.id == modelId }) {
                        // Loaded state
                        Circle()
                            .fill(Color.green)
                            .frame(width: 6, height: 6)
                        Text(model.name)
                            .font(.caption)
                            .lineLimit(1)
                            .fixedSize(horizontal: false, vertical: true)
                    } else {
                        // No model
                        Circle()
                            .fill(Color.orange)
                            .frame(width: 6, height: 6)
                        Text("Select Model")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Image(systemName: "chevron.down")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            #if os(macOS)
            .buttonStyle(.accessoryBar)
            #else
            .buttonStyle(.bordered)
            #endif
            .controlSize(.small)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .modifier(ToolbarGlassModifier())
    }

    private var emptyState: some View {
        VStack(spacing: 20) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 60))
                .foregroundStyle(.secondary)

            Text("No Conversation Selected")
                .font(.title2)
                .foregroundStyle(.secondary)

            if modelManager.loadedModelId == nil {
                VStack(spacing: 12) {
                    Text("Load a model to start chatting")
                        .foregroundStyle(.secondary)

                    Button {
                        showingModelSheet = true
                    } label: {
                        Label("Select Model", systemImage: "cpu")
                    }
                    .buttonStyle(.borderedProminent)
                }
            } else {
                Button {
                    chatVM.newConversation()
                } label: {
                    Label("New Conversation", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Conversation Row

struct ConversationRow: View {
    let conversation: Conversation

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(conversation.title)
                .font(.headline)
                .lineLimit(1)

            HStack {
                if let preview = conversation.lastMessagePreview {
                    Text(preview)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                Spacer()

                Text(conversation.formattedDate)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Conversation List Sheet (iOS)

#if os(iOS)
private struct ConversationListSheet: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(\.dismiss) private var dismiss

    let onClose: () -> Void

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(chatVM.conversations) { conversation in
                        Button {
                            chatVM.selectConversation(conversation)
                            dismiss()
                            onClose()
                        } label: {
                            ConversationRow(conversation: conversation)
                        }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button(role: .destructive) {
                                chatVM.deleteConversation(conversation)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                    .onDelete { indexSet in
                        chatVM.deleteConversation(at: indexSet)
                    }
                } header: {
                    Text("Conversations")
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Chats")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                        onClose()
                    }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        chatVM.newConversation()
                        dismiss()
                        onClose()
                    } label: {
                        Image(systemName: "plus")
                    }
                }
            }
        }
    }
}
#endif

// MARK: - Toolbar Loading Indicator

/// Compact animated loading indicator for toolbar
private struct ToolbarLoadingIndicator: View {
    let progress: ModelLoadingProgress?

    @State private var rotation: Double = 0
    @State private var pulse = false

    var body: some View {
        HStack(spacing: 4) {
            ZStack {
                // Pulsing glow
                Circle()
                    .fill(Color.green.opacity(0.3))
                    .frame(width: 16, height: 16)
                    .scaleEffect(pulse ? 1.3 : 0.9)
                    .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true), value: pulse)

                // Rotating icon
                Image(systemName: "cpu")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.green)
                    .rotationEffect(.degrees(rotation))
            }

            // Show progress detail if available, otherwise generic "Loading..."
            if let progress = progress, let detail = progress.detail, !detail.isEmpty {
                Text(detail)
                    .font(.caption2)
                    .fontWeight(.medium)
                    .foregroundStyle(.green)
            } else if let progress = progress {
                Text("\(Int(progress.percentage * 100))%")
                    .font(.caption2)
                    .fontWeight(.medium)
                    .foregroundStyle(.green)
            } else {
                Text("Loading...")
                    .font(.caption2)
                    .fontWeight(.medium)
                    .foregroundStyle(.green)
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(Color.green.opacity(0.1), in: Capsule())
        .onAppear {
            pulse = true
            withAnimation(.linear(duration: 2).repeatForever(autoreverses: false)) {
                rotation = 360
            }
        }
    }
}

// MARK: - Download Progress Pill

/// Animated download progress indicator for toolbar
private struct DownloadProgressPill: View {
    let modelManager: ModelManagerViewModel
    let onTap: () -> Void

    @State private var isAnimating = false

    private var downloadingModel: ModelInfo? {
        guard let id = modelManager.downloadingModelId else { return nil }
        return modelManager.availableModels.first { $0.id == id }
    }

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 6) {
                // Animated download icon
                ZStack {
                    Circle()
                        .fill(Color.blue.opacity(0.2))
                        .frame(width: 20, height: 20)
                        .scaleEffect(isAnimating ? 1.2 : 0.9)
                        .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isAnimating)

                    Image(systemName: "arrow.down.circle.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(.blue)
                }

                // Model name (truncated)
                if let model = downloadingModel {
                    Text(model.name)
                        .font(.caption2)
                        .fontWeight(.medium)
                        .lineLimit(1)
                        .frame(maxWidth: 80)
                }

                // Progress percentage
                if let progress = modelManager.downloadProgress {
                    Text("\(Int(progress.progress * 100))%")
                        .font(.caption2)
                        .fontWeight(.bold)
                        .foregroundStyle(.blue)
                        .monospacedDigit()
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.blue.opacity(0.1), in: Capsule())
        }
        .buttonStyle(.plain)
        .onAppear {
            isAnimating = true
        }
    }
}

// MARK: - Glass Effect Modifier (macOS 26+)

private struct ToolbarGlassModifier: ViewModifier {
    func body(content: Content) -> some View {
        #if os(macOS)
        if #available(macOS 26.0, *) {
            content
                .glassEffect(.regular)
        } else {
            content
                .background(.background)
        }
        #else
        content
            .background(.background)
        #endif
    }
}

#Preview {
    ContentView()
        .environment(ChatViewModel())
        .environment(ModelManagerViewModel())
}
