//
//  ContentView.swift
//  ANEMLLChat
//
//  Root view with navigation
//

import SwiftUI
#if os(macOS)
import AppKit
#endif

struct ContentView: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(ModelManagerViewModel.self) private var modelManager

    // Start with sidebar collapsed on macOS; keep it visible on tvOS
    @State private var columnVisibility: NavigationSplitViewVisibility = {
        #if os(macOS)
        return .detailOnly
        #else
        return .all
        #endif
    }()
    @State private var showingModelSheet = false
    @State private var showingSettings = false
    @State private var showingConversationSheet = false
    @State private var hasCheckedInitialState = false
    @State private var showingClearAllAlert = false
    @State private var tvShowsSidebar = true
    @AppStorage("largeControls") private var largeControls = StorageService.defaultLargeControlsValue

    var body: some View {
        #if os(iOS)
        iosRoot
        #elseif os(tvOS)
        tvRoot
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
        .onChange(of: modelManager.lastImportedPackageModelId) { _, importedModelId in
            guard importedModelId != nil else { return }
            showingModelSheet = true
            modelManager.lastImportedPackageModelId = nil
        }
        .onChange(of: modelManager.requestModelSelection) { _, requested in
            guard requested else { return }
            showingModelSheet = true
            modelManager.requestModelSelection = false
        }
        .overlay {
            if modelManager.isImportingIncomingPackage {
                ZStack {
                    Color.black.opacity(0.4)
                        .ignoresSafeArea()
                    Rectangle()
                        .fill(.thickMaterial)
                        .opacity(0.85)
                        .ignoresSafeArea()
                    VStack(spacing: 16) {
                        ProgressView()
                            .controlSize(.large)
                        Text(modelManager.incomingTransferStatusMessage ?? "Importing model...")
                            .font(.headline)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding(32)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                    .shadow(color: .black.opacity(0.3), radius: 20, y: 10)
                    .padding(.horizontal, 40)
                }
                .transition(.opacity)
                .allowsHitTesting(false)
            }
        }
        .toast(Binding(
            get: {
                // Only show toast when import is done (not during)
                modelManager.isImportingIncomingPackage ? nil : modelManager.incomingTransferStatusMessage
            },
            set: { modelManager.incomingTransferStatusMessage = $0 }
        ), type: .info, duration: 4)
        // Auto-show model list ONLY on true fresh start (never used before)
        .task {
            guard !hasCheckedInitialState else { return }
            hasCheckedInitialState = true

            // Check if user has ever selected a model before (indicates not a fresh install)
            let hasSelectedModelBefore = await StorageService.shared.selectedModelId != nil

            // If user has used the app before, don't auto-show model list
            if hasSelectedModelBefore {
                return
            }

            // Wait briefly for model list to load
            try? await Task.sleep(for: .milliseconds(500))

            // Only show on true fresh start: no downloaded models AND no previous selection
            if modelManager.downloadedModels.isEmpty && !modelManager.isLoadingModel {
                showingModelSheet = true
            }
        }
    }
    #elseif os(tvOS)
    private var tvRoot: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.04, green: 0.05, blue: 0.06),
                    Color(red: 0.02, green: 0.02, blue: 0.03)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            HStack(spacing: 0) {
                if tvShowsSidebar {
                    tvSidebar
                        .frame(width: 360)
                        .focusSection()
                        .transition(.move(edge: .leading).combined(with: .opacity))
                }

                tvDetail
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .focusSection()
                    // Intercept Menu/Back button:
                    // Always toggle sidebar — do NOT cancel inference.
                    // User can stop generation with the Stop button in InputBar.
                    .onExitCommand {
                        if !tvShowsSidebar {
                            withAnimation(.easeInOut(duration: 0.22)) {
                                tvShowsSidebar = true
                            }
                        }
                    }
            }
            .animation(.easeInOut(duration: 0.22), value: tvShowsSidebar)
        }
        .fullScreenCover(isPresented: $showingModelSheet) {
            TVFullscreenDialog(showsCloseButton: false) {
                ModelListView()
                    .environment(modelManager)
                    .environment(chatVM)
            }
        }
        .fullScreenCover(isPresented: $showingConversationSheet) {
            TVConversationDialog {
                showTVDetail()
            }
                .environment(chatVM)
        }
        .fullScreenCover(isPresented: $showingSettings) {
            TVFullscreenDialog(showsCloseButton: false) {
                SettingsView()
                    .environment(chatVM)
                    #if os(tvOS)
                    .environment(modelManager)
                    #endif
            }
        }
        .onChange(of: modelManager.lastImportedPackageModelId) { _, importedModelId in
            guard importedModelId != nil else { return }
            showingModelSheet = true
            modelManager.lastImportedPackageModelId = nil
        }
        .onChange(of: modelManager.requestModelSelection) { _, requested in
            guard requested else { return }
            showingModelSheet = true
            modelManager.requestModelSelection = false
        }
        .overlay {
            if modelManager.isImportingIncomingPackage {
                ZStack {
                    Color.black.opacity(0.4)
                        .ignoresSafeArea()
                    Rectangle()
                        .fill(.thickMaterial)
                        .opacity(0.85)
                        .ignoresSafeArea()
                    VStack(spacing: 16) {
                        ProgressView()
                            .controlSize(.large)
                        Text(modelManager.incomingTransferStatusMessage ?? "Importing model...")
                            .font(.headline)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding(32)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                    .shadow(color: .black.opacity(0.3), radius: 20, y: 10)
                    .padding(.horizontal, 40)
                }
                .transition(.opacity)
                .allowsHitTesting(false)
            }
        }
        .toast(Binding(
            get: {
                modelManager.isImportingIncomingPackage ? nil : modelManager.incomingTransferStatusMessage
            },
            set: { modelManager.incomingTransferStatusMessage = $0 }
        ), type: .info, duration: 4)
        .task {
            guard !hasCheckedInitialState else { return }
            hasCheckedInitialState = true

            let hasSelectedModelBefore = await StorageService.shared.selectedModelId != nil
            if hasSelectedModelBefore {
                return
            }

            try? await Task.sleep(for: .milliseconds(500))

            if modelManager.downloadedModels.isEmpty && !modelManager.isLoadingModel {
                showingModelSheet = true
            }
        }
        .onChange(of: chatVM.currentConversation?.id) { _, conversationId in
            guard conversationId != nil else { return }
            showTVDetail()
        }
    }

    private var tvSidebar: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Title with stylish font
            Text("ANEMLL Chat")
                .font(.system(size: 32, weight: .bold, design: .rounded))
                .foregroundStyle(
                    LinearGradient(
                        colors: [.white, .white.opacity(0.7)],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .padding(.bottom, 4)

            // All buttons uniform width
            VStack(spacing: 14) {
                Button {
                    startTVNewChat()
                } label: {
                    Label("New Chat", systemImage: "plus.bubble")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

                Button {
                    showingModelSheet = true
                } label: {
                    Label(loadedModelName ?? "Select Model", systemImage: "cpu")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)

                Button {
                    showingConversationSheet = true
                } label: {
                    Label("Chats", systemImage: "text.bubble")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)

                Button {
                    showingSettings = true
                } label: {
                    Label("Settings", systemImage: "gearshape")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }

            if !chatVM.conversations.isEmpty {
                Divider()
                    .overlay(Color.white.opacity(0.12))
                    .padding(.vertical, 2)
                Text("\(chatVM.conversations.count) saved chats")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 0)

            // App icon at bottom
            Image("AppIcon")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 48, height: 48)
                .clipShape(RoundedRectangle(cornerRadius: 11, style: .continuous))
                .opacity(0.7)
                .padding(.bottom, 4)
        }
        .padding(.horizontal, 28)
        .padding(.vertical, 28)
        .background(
            ZStack {
                LinearGradient(
                    colors: [
                        Color.black.opacity(0.42),
                        Color.black.opacity(0.28)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                // Glass-like subtle inner glow
                RoundedRectangle(cornerRadius: 0)
                    .stroke(Color.white.opacity(0.06), lineWidth: 1)
            }
        )
    }

    private var tvDetail: some View {
        VStack(spacing: 0) {
            tvTopBar
                .focusSection()
            tvModelStatusPanel
            ChatView()
                .environment(chatVM)
                .environment(modelManager)
        }
    }

    private var tvTopBar: some View {
        HStack(spacing: 14) {
            Button {
                toggleTVSidebar()
            } label: {
                Label(tvShowsSidebar ? "Hide Menu" : "Menu", systemImage: "sidebar.leading")
            }
            .buttonStyle(.bordered)

            Spacer()

            if let loadedModelName {
                Label(loadedModelName, systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .font(.subheadline)
            } else if modelManager.isLoadingModel {
                // Show a brief loading indicator instead of "No model loaded"
                HStack(spacing: 8) {
                    ProgressView()
                    Text(modelManager.loadingModelName ?? "Loading...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            } else {
                Label("No model loaded", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .font(.subheadline)
            }

            if modelManager.downloadingModelId != nil {
                Label("Downloading", systemImage: "arrow.down.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }

    private var tvModelStatusPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Model loading progress is shown by the centered gauge in ChatView,
            // so skip the duplicate top-left panel during loading
            if modelManager.isLoadingModel {
                EmptyView()
            } else if let progress = modelManager.downloadProgress,
                      modelManager.downloadingModelId != nil {
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Label("Downloading model", systemImage: "arrow.down.circle.fill")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    Spacer()
                    Text(progress.progressPercent)
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }

                ProgressView(value: progress.progress)
                    .progressViewStyle(.linear)

                Text("\(progress.downloadedString) / \(progress.totalString) • \(progress.speedString)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if let error = modelManager.errorMessage,
                      !error.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                }
            } else if modelManager.loadedModelId == nil {
                HStack(spacing: 10) {
                    Label("No model loaded", systemImage: "cpu")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Open Models") {
                        showingModelSheet = true
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(Color.black.opacity(0.22))
        .overlay(
            Rectangle()
                .fill(Color.white.opacity(0.08))
                .frame(height: 1),
            alignment: .bottom
        )
    }

    private func startTVNewChat() {
        chatVM.newConversation()
        showTVDetail()
    }

    private func showTVDetail() {
        withAnimation(.easeInOut(duration: 0.2)) {
            tvShowsSidebar = false
        }
    }

    private func toggleTVSidebar() {
        withAnimation(.easeInOut(duration: 0.2)) {
            tvShowsSidebar.toggle()
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
                .frame(minWidth: 720, maxWidth: .infinity, minHeight: 560, maxHeight: .infinity)
                .onResolveWindow { window in
                    window.styleMask.insert(.resizable)
                    window.minSize = NSSize(width: 720, height: 560)
                }
        }
        .onChange(of: modelManager.lastImportedPackageModelId) { importedModelId in
            guard importedModelId != nil else { return }
            showingModelSheet = true
            modelManager.lastImportedPackageModelId = nil
        }
        .onChange(of: modelManager.requestModelSelection) { _, requested in
            guard requested else { return }
            showingModelSheet = true
            modelManager.requestModelSelection = false
        }
        .overlay {
            if modelManager.isImportingIncomingPackage {
                ZStack {
                    Color.black.opacity(0.4)
                        .ignoresSafeArea()
                    Rectangle()
                        .fill(.thickMaterial)
                        .opacity(0.85)
                        .ignoresSafeArea()
                    VStack(spacing: 16) {
                        ProgressView()
                            .controlSize(.large)
                        Text(modelManager.incomingTransferStatusMessage ?? "Importing model...")
                            .font(.headline)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding(32)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                    .shadow(color: .black.opacity(0.3), radius: 20, y: 10)
                    .padding(.horizontal, 40)
                }
                .transition(.opacity)
                .allowsHitTesting(false)
            }
        }
        .toast(Binding(
            get: {
                modelManager.isImportingIncomingPackage ? nil : modelManager.incomingTransferStatusMessage
            },
            set: { modelManager.incomingTransferStatusMessage = $0 }
        ), type: .info, duration: 4)
        // Auto-show model list ONLY on true fresh start (never used before)
        .task {
            guard !hasCheckedInitialState else { return }
            hasCheckedInitialState = true

            // Check if user has ever selected a model before (indicates not a fresh install)
            let hasSelectedModelBefore = await StorageService.shared.selectedModelId != nil

            // If user has used the app before, don't auto-show model list
            if hasSelectedModelBefore {
                return
            }

            // Wait briefly for model list to load
            try? await Task.sleep(for: .milliseconds(500))

            // Only show on true fresh start: no downloaded models AND no previous selection
            if modelManager.downloadedModels.isEmpty && !modelManager.isLoadingModel {
                showingModelSheet = true
            }
        }
    }
    #endif

    private var loadedModelName: String? {
        guard let id = modelManager.loadedModelId else { return nil }
        return modelManager.availableModels.first(where: { $0.id == id })?.name
    }

    // MARK: - Sidebar

    #if !os(tvOS)
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
    #endif

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
        let iconSize: CGFloat = largeControls ? 28 : 14
        let statusDotSize: CGFloat = largeControls ? 12 : 6
        let horizontalPadding: CGFloat = largeControls ? 20 : 12
        let verticalPadding: CGFloat = largeControls ? 14 : 8
        let spacing: CGFloat = largeControls ? 18 : 10

        return HStack {
            Spacer()
            HStack(spacing: spacing) {
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
                            .frame(width: statusDotSize, height: statusDotSize)
                            .offset(x: statusDotSize / 2, y: statusDotSize / 2)
                    }
                }

                Button {
                    showingSettings = true
                } label: {
                    Image(systemName: "gearshape")
                }
            }
            .font(.system(size: iconSize, weight: .semibold))
            .foregroundStyle(.primary)
            .padding(.horizontal, horizontalPadding)
            .padding(.vertical, verticalPadding)
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

            #if os(macOS)
            // Settings button (opens standard macOS Settings scene)
            SettingsLink {
                Image(systemName: "gearshape")
            }
            .buttonStyle(.accessoryBar)
            .controlSize(.small)
            .help("Settings (⌘,)")
            #endif

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
        VStack(spacing: 16) {
            Spacer()

            // App icon
            #if os(macOS)
            Image(nsImage: NSApp.applicationIconImage)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 96, height: 96)
            #else
            Image("AppIcon")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 96, height: 96)
                .clipShape(RoundedRectangle(cornerRadius: 22))
            #endif

            Text("ANEMLL Chat")
                .font(.title)
                .fontWeight(.semibold)
                .foregroundStyle(.primary)

            // Action area
            Text("Start a new conversation")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Button {
                #if os(tvOS)
                startTVNewChat()
                #else
                chatVM.newConversation()
                #endif
            } label: {
                Label("New Conversation", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

#if os(macOS)
private struct WindowResolver: NSViewRepresentable {
    let onResolve: (NSWindow) -> Void

    func makeNSView(context: Context) -> NSView {
        NSView(frame: .zero)
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async {
            guard let window = nsView.window else { return }
            onResolve(window)
        }
    }
}

private extension View {
    func onResolveWindow(_ onResolve: @escaping (NSWindow) -> Void) -> some View {
        background(WindowResolver(onResolve: onResolve).frame(width: 0, height: 0))
    }
}
#endif

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

#if os(tvOS)
private struct TVFullscreenDialog<Content: View>: View {
    @Environment(\.dismiss) private var dismiss
    let showsCloseButton: Bool
    let content: Content

    init(showsCloseButton: Bool = true, @ViewBuilder content: () -> Content) {
        self.showsCloseButton = showsCloseButton
        self.content = content()
    }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            LinearGradient(
                colors: [
                    Color.black.opacity(0.96),
                    Color(red: 0.06, green: 0.06, blue: 0.08)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            if showsCloseButton {
                Button("Close") {
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .padding(.top, 24)
                .padding(.trailing, 32)
            }
        }
        // Menu button on remote dismisses fullscreen dialogs
        .onExitCommand {
            dismiss()
        }
    }
}

private struct TVConversationDialog: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(\.dismiss) private var dismiss

    @State private var showingClearAlert = false
    let onOpenChat: () -> Void

    var body: some View {
        TVFullscreenDialog {
            VStack(spacing: 16) {
                HStack {
                    Text("Conversations")
                        .font(.title2)
                        .fontWeight(.bold)
                    Spacer()
                }

                if chatVM.conversations.isEmpty {
                    VStack(spacing: 10) {
                        Image(systemName: "text.bubble")
                            .font(.system(size: 36))
                            .foregroundStyle(.secondary)
                        Text("No conversations yet")
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(chatVM.conversations) { conversation in
                        Button {
                            chatVM.selectConversation(conversation)
                            onOpenChat()
                            dismiss()
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
                    .listStyle(.plain)
                }

                HStack(spacing: 12) {
                    Button {
                        chatVM.newConversation()
                        onOpenChat()
                        dismiss()
                    } label: {
                        Label("New Chat", systemImage: "plus")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)

                    Button(role: .destructive) {
                        showingClearAlert = true
                    } label: {
                        Label("Clear All", systemImage: "trash")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .disabled(chatVM.conversations.isEmpty)

                    Spacer()
                }
            }
            .padding(.horizontal, 42)
            .padding(.top, 36)
            .padding(.bottom, 28)
        }
        .alert("Clear All Chats?", isPresented: $showingClearAlert) {
            Button("Cancel", role: .cancel) {}
            Button("Clear All", role: .destructive) {
                chatVM.clearAllConversations()
            }
        } message: {
            Text("This will delete all conversations. This action cannot be undone.")
        }
    }
}
#endif

// MARK: - Conversation List Sheet (iOS)

#if os(iOS)
private struct ConversationListSheet: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(\.dismiss) private var dismiss

    @State private var showingClearAlert = false

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
                    HStack(spacing: 16) {
                        // Clear all button
                        Button(role: .destructive) {
                            showingClearAlert = true
                        } label: {
                            Image(systemName: "trash")
                        }
                        .disabled(chatVM.conversations.isEmpty)

                        // New chat button
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
            .alert("Clear All Chats?", isPresented: $showingClearAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Clear All", role: .destructive) {
                    chatVM.clearAllConversations()
                }
            } message: {
                Text("This will delete all conversations. This action cannot be undone.")
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

                // Progress percentage or preparing state
                if let progress = modelManager.downloadProgress {
                    Text("\(Int(progress.progress * 100))%")
                        .font(.caption2)
                        .fontWeight(.bold)
                        .foregroundStyle(.blue)
                        .monospacedDigit()
                } else {
                    Text("Preparing...")
                        .font(.caption2)
                        .foregroundStyle(.blue.opacity(0.7))
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
