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

    @State private var columnVisibility: NavigationSplitViewVisibility = .all
    @State private var showingModelSheet = false
    @State private var showingSettings = false

    var body: some View {
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
        #if os(iOS)
        .sheet(isPresented: $showingSettings) {
            NavigationStack {
                SettingsView()
                    .environment(chatVM)
            }
        }
        #endif
        .task {
            await modelManager.autoLoadLastModel()
        }
    }

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

    private var detailToolbar: some View {
        HStack {
            // New Chat button
            Button {
                chatVM.newConversation()
            } label: {
                Label("New Chat", systemImage: "plus.bubble")
            }
            .buttonStyle(.bordered)

            Spacer()

            // Model loading indicator
            if modelManager.isLoadingModel {
                HStack(spacing: 6) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Loading model...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Models button
            Button {
                showingModelSheet = true
            } label: {
                HStack(spacing: 6) {
                    Circle()
                        .fill(modelManager.loadedModelId != nil ? Color.green : Color.orange)
                        .frame(width: 8, height: 8)
                    if let modelId = modelManager.loadedModelId,
                       let model = modelManager.availableModels.first(where: { $0.id == modelId }) {
                        Text(model.name)
                            .lineLimit(1)
                    } else {
                        Text("No Model")
                            .foregroundStyle(.secondary)
                    }
                    Image(systemName: "chevron.down")
                        .font(.caption2)
                }
            }
            .buttonStyle(.bordered)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(NSColor.windowBackgroundColor))
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

#Preview {
    ContentView()
        .environment(ChatViewModel())
        .environment(ModelManagerViewModel())
}
