//
//  SettingsView.swift
//  ANEMLLChat
//
//  App settings and configuration
//

import SwiftUI
import UniformTypeIdentifiers

// System prompt options
enum SystemPromptOption: String, CaseIterable, Identifiable {
    case defaultPrompt = "Default Prompt"       // Basic inference with no additional prompting (DEFAULT)
    case noTemplate = "No Template"             // Raw inference without chat template
    case modelThinking = "Thinking Mode"        // Model's thinking/reasoning mode if supported
    case modelNonThinking = "Non-Thinking Mode" // Model's non-thinking mode if supported
    case custom = "Custom"                      // User-defined system prompt

    var id: String { rawValue }
}

struct SettingsView: View {
    @Environment(ChatViewModel.self) private var chatVM
    #if os(macOS)
    @Environment(ModelManagerViewModel.self) private var modelManager
    #endif
    @Environment(\.dismiss) private var dismiss

    @State private var temperature: Float = 0.0
    @State private var maxTokens: Int = 2048
    @State private var systemPromptOption: SystemPromptOption = .defaultPrompt
    @State private var customPrompt: String = ""

    @State private var showingLogs = false
    @State private var autoLoadLastModel = true
    @State private var debugLevel: Int = 0
    @State private var repetitionDetectionEnabled = false
    @State private var enableMarkup = StorageService.defaultEnableMarkupValue
    @State private var sendButtonOnLeft = StorageService.defaultSendButtonOnLeftValue
    @State private var loadLastChat = StorageService.defaultLoadLastChatValue
    @State private var largeControls = StorageService.defaultLargeControlsValue
    @State private var showMicrophone = StorageService.defaultShowMicrophoneValue
    @State private var showingResetConfirmation = false
    #if os(macOS)
    @State private var macOSStorageFolderPath = ""
    @State private var showingStorageFolderPicker = false
    @State private var pendingStorageMigration: (oldURL: URL, newURL: URL)?
    @State private var showingStorageMigrationPrompt = false
    @State private var showingStorageMigrationOptions = false
    @State private var storageFolderStatusMessage: String?
    @State private var isStorageMigrationInProgress = false
    @State private var storageMigrationProgressValue: Double = 0
    @State private var storageMigrationProgressMessage = ""
    #endif

    var body: some View {
        Form {
            // Model settings
            modelSection

            // Generation settings
            generationSection

            // Display settings
            displaySection

            // System prompt
            systemPromptSection

            // Logs
            logsSection

            // About
            aboutSection
        }
        .formStyle(.grouped)
        .navigationTitle("Settings")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Done") {
                    saveSettings()
                    dismiss()
                }
            }
        }
        #endif
        .onAppear {
            loadSettings()
        }
        .onDisappear {
            // Save settings when view closes (especially important for macOS which has no Done button)
            saveSettings()
        }
        .sheet(isPresented: $showingLogs) {
            LogsView()
        }
        #if os(macOS)
        .fileImporter(
            isPresented: $showingStorageFolderPicker,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let selected = urls.first else { return }
                handleStorageFolderSelection(selected)
            case .failure(let error):
                storageFolderStatusMessage = "Failed to select folder: \(error.localizedDescription)"
            }
        }
        .alert("Migrate Existing Data?", isPresented: $showingStorageMigrationPrompt) {
            Button("Not Now", role: .cancel) {
                pendingStorageMigration = nil
            }
            Button("Yes") {
                showingStorageMigrationOptions = true
            }
        } message: {
            if let migration = pendingStorageMigration {
                Text("Storage changed from \(migration.oldURL.path) to \(migration.newURL.path). Migrate existing chats and models to the new folder?")
            } else {
                Text("Migrate existing chats and models to the new folder?")
            }
        }
        .confirmationDialog("Migrate Storage Data", isPresented: $showingStorageMigrationOptions, titleVisibility: .visible) {
            Button("Copy Files") {
                performStorageMigration(.copy)
            }
            Button("Move Files") {
                performStorageMigration(.move)
            }
            Button("Cancel", role: .cancel) {
                pendingStorageMigration = nil
            }
        } message: {
            Text("Choose how to migrate existing data. Only app data under this folder is migrated; the source root folder itself is never moved.")
        }
        #endif
    }

    // MARK: - Model Section

    private var modelSection: some View {
        Section {
            Toggle("Auto-load last model", isOn: $autoLoadLastModel)
            Toggle("Load last chat on startup", isOn: $loadLastChat)

            Button(role: .destructive) {
                Task {
                    await StorageService.shared.clearLastModel()
                }
            } label: {
                Label("Clear remembered model", systemImage: "xmark.circle")
            }

            #if os(macOS)
            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Text("Storage Folder")
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(macOSStorageFolderPath)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .lineLimit(2)

                Button {
                    showingStorageFolderPicker = true
                } label: {
                    Label("Change Storage Folder", systemImage: "folder")
                }
                .disabled(isStorageMigrationInProgress)
            }

            if let storageFolderStatusMessage {
                Text(storageFolderStatusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if isStorageMigrationInProgress {
                VStack(alignment: .leading, spacing: 6) {
                    ProgressView(value: storageMigrationProgressValue)
                    Text(storageMigrationProgressMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            #endif
        } header: {
            Text("Model")
        } footer: {
            #if os(macOS)
            Text(loadLastChat ? "App will restore your last conversation on startup. Storage folder applies to macOS only." : "App will start with a new chat on startup. Storage folder applies to macOS only.")
            #else
            Text(loadLastChat ? "App will restore your last conversation on startup" : "App will start with a new chat on startup")
            #endif
        }
    }

    // MARK: - Generation Section

    private var generationSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Temperature")
                    Spacer()
                    Text(String(format: "%.2f", temperature))
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }

                Slider(value: $temperature, in: 0...2, step: 0.05)

                Text("Lower = more focused, Higher = more creative")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Max Tokens")
                    Spacer()
                    Text("\(maxTokens)")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }

                Slider(
                    value: Binding(
                        get: { Double(maxTokens) },
                        set: { maxTokens = Int($0) }
                    ),
                    in: 64...2048,
                    step: 64
                )

                Text("Maximum number of tokens to generate")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Toggle("Repetition Detection", isOn: $repetitionDetectionEnabled)
        } header: {
            Text("Generation")
        } footer: {
            Text(repetitionDetectionEnabled ? "Stops generation if repetitive patterns are detected" : "Generation continues until EOS or max tokens (CLI behavior)")
        }
    }

    // MARK: - System Prompt Section

    private var systemPromptSection: some View {
        Section {
            Picker("Prompt", selection: $systemPromptOption) {
                ForEach(SystemPromptOption.allCases) { option in
                    Text(option.rawValue).tag(option)
                }
            }

            if systemPromptOption == .custom {
                TextEditor(text: $customPrompt)
                    .frame(minHeight: 80)
                    .font(.body)
            }
        } header: {
            Text("System Prompt")
        } footer: {
            switch systemPromptOption {
            case .defaultPrompt:
                Text("Standard inference with chat template, no additional system prompt")
            case .noTemplate:
                Text("Raw inference without chat template - direct model output")
            case .modelThinking:
                Text("Uses thinking/reasoning mode if supported by the model")
            case .modelNonThinking:
                Text("Uses non-thinking mode if supported by the model")
            case .custom:
                Text("Custom system prompt instructions for the AI")
            }
        }
    }

    // MARK: - Display Section

    private var displaySection: some View {
        Section {
            Toggle("Enable Markup", isOn: $enableMarkup)
            Toggle("Send Button on Left", isOn: $sendButtonOnLeft)
            Toggle("Show Microphone", isOn: $showMicrophone)
            #if os(iOS) || os(visionOS)
            Toggle("Large Controls", isOn: $largeControls)
            #endif
        } header: {
            Text("Display")
        } footer: {
            #if os(iOS) || os(visionOS)
            Text(largeControls ? "Send button and toolbar icons are enlarged for easier touch" : "Standard control sizes")
            #else
            Text(showMicrophone ? "Voice input button is shown next to the text field" : "Voice input button is hidden")
            #endif
        }
    }

    // MARK: - Logs Section

    private var logsSection: some View {
        Section {
            Picker("Debug Level", selection: $debugLevel) {
                Text("Off").tag(0)
                Text("Basic").tag(1)
                Text("Verbose").tag(2)
            }

            Button {
                showingLogs = true
            } label: {
                HStack {
                    Label("View Logs", systemImage: "doc.text")
                    Spacer()
                    Image(systemName: "chevron.right")
                        .foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)
        } header: {
            Text("Debug")
        } footer: {
            Text("Debug level affects console output during model loading and inference")
        }
    }

    // MARK: - About Section

    private var aboutSection: some View {
        Section {
            HStack {
                Text("Version")
                Spacer()
                Text("1.0.0")
                    .foregroundStyle(.secondary)
            }

            HStack {
                Text("Build")
                Spacer()
                Text(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "1")
                    .foregroundStyle(.secondary)
            }

            Link(destination: URL(string: "https://github.com/anemll/anemll")!) {
                HStack {
                    Label("GitHub", systemImage: "link")
                    Spacer()
                    Image(systemName: "arrow.up.right")
                        .foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)

            Link(destination: URL(string: "https://huggingface.co/anemll")!) {
                HStack {
                    Label("HuggingFace Models", systemImage: "link")
                    Spacer()
                    Image(systemName: "arrow.up.right")
                        .foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)

            Button(role: .destructive) {
                showingResetConfirmation = true
            } label: {
                Label("Reset to Defaults", systemImage: "arrow.counterclockwise")
            }
            .confirmationDialog("Reset all settings to defaults?", isPresented: $showingResetConfirmation, titleVisibility: .visible) {
                Button("Reset", role: .destructive) {
                    resetToDefaults()
                }
                Button("Cancel", role: .cancel) {}
            }
        } header: {
            Text("About")
        } footer: {
            Text("ANEMLL Chat - On-device LLM inference powered by Apple Neural Engine")
        }
    }

    // MARK: - Actions

    private func loadSettings() {
        temperature = chatVM.temperature
        maxTokens = chatVM.maxTokens

        // Parse the stored system prompt to determine option
        let storedPrompt = chatVM.systemPrompt
        if storedPrompt.isEmpty || storedPrompt == "[DEFAULT_PROMPT]" {
            systemPromptOption = .defaultPrompt
        } else if storedPrompt == "[NO_TEMPLATE]" {
            systemPromptOption = .noTemplate
        } else if storedPrompt.hasPrefix("[MODEL_THINKING]") {
            systemPromptOption = .modelThinking
        } else if storedPrompt.hasPrefix("[MODEL_NON_THINKING]") {
            systemPromptOption = .modelNonThinking
        } else if storedPrompt.hasPrefix("[MODEL_DEFAULT]") {
            // Legacy: treat old MODEL_DEFAULT as defaultPrompt
            systemPromptOption = .defaultPrompt
        } else {
            systemPromptOption = .custom
            customPrompt = storedPrompt
        }

        Task {
            autoLoadLastModel = await StorageService.shared.autoLoadLastModel
            debugLevel = await StorageService.shared.debugLevel
            repetitionDetectionEnabled = await StorageService.shared.repetitionDetectionEnabled
            enableMarkup = await StorageService.shared.enableMarkup
            sendButtonOnLeft = await StorageService.shared.sendButtonOnLeft
            loadLastChat = await StorageService.shared.loadLastChat
            largeControls = await StorageService.shared.largeControls
            showMicrophone = await StorageService.shared.showMicrophone
            #if os(macOS)
            macOSStorageFolderPath = await StorageService.shared.currentMacOSStorageFolderURL().path
            #endif
        }
    }

    private func saveSettings() {
        chatVM.temperature = temperature
        chatVM.maxTokens = maxTokens

        // Convert option to stored string
        switch systemPromptOption {
        case .defaultPrompt:
            chatVM.systemPrompt = "[DEFAULT_PROMPT]"
        case .noTemplate:
            chatVM.systemPrompt = "[NO_TEMPLATE]"
        case .modelThinking:
            chatVM.systemPrompt = "[MODEL_THINKING]"
        case .modelNonThinking:
            chatVM.systemPrompt = "[MODEL_NON_THINKING]"
        case .custom:
            chatVM.systemPrompt = customPrompt
        }

        Task {
            await chatVM.saveSettings()
            await StorageService.shared.saveAutoLoadLastModel(autoLoadLastModel)
            await StorageService.shared.saveDebugLevel(debugLevel)
            await StorageService.shared.saveRepetitionDetectionEnabled(repetitionDetectionEnabled)
            await StorageService.shared.saveEnableMarkup(enableMarkup)
            await StorageService.shared.saveSendButtonOnLeft(sendButtonOnLeft)
            await StorageService.shared.saveLoadLastChat(loadLastChat)
            await StorageService.shared.saveLargeControls(largeControls)
            await StorageService.shared.saveShowMicrophone(showMicrophone)
            // Update InferenceService settings
            await MainActor.run {
                InferenceService.shared.debugLevel = debugLevel
                InferenceService.shared.repetitionDetectionEnabled = repetitionDetectionEnabled
            }
        }
    }

    private func resetToDefaults() {
        // Reset local state to defaults
        temperature = StorageService.defaultTemperatureValue
        maxTokens = StorageService.defaultMaxTokensValue
        systemPromptOption = .defaultPrompt  // Default Prompt is the default setting
        customPrompt = ""
        autoLoadLastModel = StorageService.defaultAutoLoadLastModelValue
        debugLevel = StorageService.defaultDebugLevelValue
        repetitionDetectionEnabled = StorageService.defaultRepetitionDetectionValue
        enableMarkup = StorageService.defaultEnableMarkupValue
        sendButtonOnLeft = StorageService.defaultSendButtonOnLeftValue
        loadLastChat = StorageService.defaultLoadLastChatValue
        largeControls = StorageService.defaultLargeControlsValue
        showMicrophone = StorageService.defaultShowMicrophoneValue

        // Save to storage
        Task {
            await StorageService.shared.resetToDefaults()
            // Update view model
            chatVM.temperature = temperature
            chatVM.maxTokens = maxTokens
            chatVM.systemPrompt = StorageService.defaultSystemPromptValue
            await chatVM.saveSettings()
            // Update InferenceService
            await MainActor.run {
                InferenceService.shared.debugLevel = debugLevel
                InferenceService.shared.repetitionDetectionEnabled = repetitionDetectionEnabled
            }
        }
    }

    #if os(macOS)
    private func handleStorageFolderSelection(_ selectedURL: URL) {
        Task {
            do {
                let update = try await StorageService.shared.updateMacOSStorageFolder(to: selectedURL)

                await MainActor.run {
                    macOSStorageFolderPath = update.newURL.path
                }

                guard update.changed else {
                    await MainActor.run {
                        storageFolderStatusMessage = "Storage folder is unchanged."
                    }
                    return
                }

                await refreshStorageBackedData()

                await MainActor.run {
                    pendingStorageMigration = (oldURL: update.oldURL, newURL: update.newURL)
                    storageFolderStatusMessage = "Storage folder changed. Choose whether to migrate existing data."
                    showingStorageMigrationPrompt = true
                }

            } catch {
                await MainActor.run {
                    storageFolderStatusMessage = "Failed to change storage folder: \(error.localizedDescription)"
                }
            }
        }
    }

    private func performStorageMigration(_ mode: StorageMigrationMode) {
        guard let migration = pendingStorageMigration else { return }

        Task {
            await MainActor.run {
                isStorageMigrationInProgress = true
                storageMigrationProgressValue = 0
                storageMigrationProgressMessage = mode == .copy ? "Preparing copy..." : "Preparing move..."
                storageFolderStatusMessage = nil
            }

            do {
                try await StorageService.shared.migrateMacOSStorage(
                    from: migration.oldURL,
                    to: migration.newURL,
                    mode: mode,
                    progress: { progress in
                        Task { @MainActor in
                            storageMigrationProgressValue = progress.fractionCompleted
                            storageMigrationProgressMessage = progress.message
                        }
                    }
                )
                await refreshStorageBackedData()
                await MainActor.run {
                    isStorageMigrationInProgress = false
                    storageMigrationProgressValue = 1.0
                    storageMigrationProgressMessage = mode == .copy ? "Copy complete." : "Move complete."
                    storageFolderStatusMessage = mode == .copy
                        ? "Copied existing data to the new storage folder. Source root folder was left unchanged."
                        : "Moved app data to the new storage folder. Source root folder was left unchanged."
                    pendingStorageMigration = nil
                }
            } catch {
                await MainActor.run {
                    isStorageMigrationInProgress = false
                    storageFolderStatusMessage = "Migration failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func refreshStorageBackedData() async {
        await chatVM.loadConversations()
        await modelManager.loadModels()
    }
    #endif
}

// MARK: - Logs View

struct LogsView: View {
    @Environment(\.dismiss) private var dismiss

    @State private var logs: [AppLogger.LogEntry] = []
    @State private var selectedLevel: LogLevel? = nil

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Filter
                Picker("Level", selection: $selectedLevel) {
                    Text("All").tag(nil as LogLevel?)
                    ForEach([LogLevel.debug, .info, .warning, .error], id: \.self) { level in
                        Text(level.emoji).tag(level as LogLevel?)
                    }
                }
                .pickerStyle(.segmented)
                .padding()

                // Logs list
                List(filteredLogs) { entry in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(entry.level.emoji)
                            Text(entry.formattedTimestamp)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text("[\(entry.category.rawValue)]")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }

                        Text(entry.message)
                            .font(.caption)
                            .textSelection(.enabled)
                    }
                }
                .listStyle(.plain)
            }
            .navigationTitle("Logs")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .primaryAction) {
                    Menu {
                        Button {
                            copyLogs()
                        } label: {
                            Label("Copy All", systemImage: "doc.on.doc")
                        }

                        Button(role: .destructive) {
                            clearLogs()
                        } label: {
                            Label("Clear", systemImage: "trash")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .onAppear {
                loadLogs()
            }
        }
    }

    private var filteredLogs: [AppLogger.LogEntry] {
        if let level = selectedLevel {
            return logs.filter { $0.level == level }
        }
        return logs
    }

    private func loadLogs() {
        logs = AppLogger.shared.recentLogs.reversed()
    }

    private func copyLogs() {
        let text = AppLogger.shared.exportLogs()
        #if os(iOS)
        UIPasteboard.general.string = text
        #else
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #endif
    }

    private func clearLogs() {
        AppLogger.shared.clearLogs()
        logs = []
    }
}

#Preview {
    NavigationStack {
        SettingsView()
            .environment(ChatViewModel())
    }
}
