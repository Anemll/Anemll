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

    // Sampling settings
    @State private var doSample: Bool = false
    @State private var topP: Float = 0.95
    @State private var topK: Int = 0
    @State private var useRecommendedSampling: Bool = true

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
    @State private var showingAcknowledgements = false
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
        .sheet(isPresented: $showingAcknowledgements) {
            AcknowledgementsView()
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
            // Sampling toggle and recommended sampling
            samplingControls

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Temperature")
                    Spacer()
                    Text(String(format: "%.2f", temperature))
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }

                Slider(value: $temperature, in: 0...2, step: 0.05)
                    .disabled(useRecommendedSampling && hasRecommendedSampling)

                Text("Lower = more focused, Higher = more creative")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Top-P and Top-K controls (only when sampling is enabled)
            if doSample || (useRecommendedSampling && hasRecommendedSampling) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Top-P")
                        Spacer()
                        Text(String(format: "%.2f", topP))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }

                    Slider(value: $topP, in: 0...1, step: 0.05)
                        .disabled(useRecommendedSampling && hasRecommendedSampling)

                    Text("Nucleus sampling threshold")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Top-K")
                        Spacer()
                        Text(topK == 0 ? "Off" : "\(topK)")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }

                    Slider(
                        value: Binding(
                            get: { Double(topK) },
                            set: { topK = Int($0) }
                        ),
                        in: 0...100,
                        step: 5
                    )
                    .disabled(useRecommendedSampling && hasRecommendedSampling)

                    Text("Top-K sampling (0 = disabled)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
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
                    in: 64...Double(maxTokensLimit),
                    step: 64
                )

                Text("Maximum number of tokens to generate (model max: \(maxTokensLimit))")
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

    // MARK: - Sampling Controls

    private var hasRecommendedSampling: Bool {
        InferenceService.shared.modelRecommendedSampling != nil
    }

    private var isArgmaxModel: Bool {
        InferenceService.shared.isArgmaxModel
    }

    /// Maximum tokens limit based on current model's context size
    private var maxTokensLimit: Int {
        InferenceService.shared.modelMaxContextSize
    }

    @ViewBuilder
    private var samplingControls: some View {
        // Always show this toggle - it's a global preference
        Toggle("Use Model Sampling (if available)", isOn: $useRecommendedSampling)
            .onChange(of: useRecommendedSampling) { _, newValue in
                if newValue, let rec = InferenceService.shared.modelRecommendedSampling {
                    // Apply recommended values
                    doSample = rec.doSample
                    temperature = rec.temperature
                    topP = rec.topP
                    topK = rec.topK
                }
            }

        // Show status based on current model
        if isArgmaxModel {
            // Argmax model - sampling unavailable
            HStack {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                Text("Sampling unavailable (argmax model)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } else if useRecommendedSampling && hasRecommendedSampling {
            // Model has recommendations and user wants to use them
            if let rec = InferenceService.shared.modelRecommendedSampling {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text("Using: \(String(format: "%.2f", rec.temperature)) / \(String(format: "%.2f", rec.topP)) / \(rec.topK)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        } else if useRecommendedSampling && !hasRecommendedSampling {
            // User wants recommendations but model doesn't have any
            HStack {
                Image(systemName: "info.circle")
                    .foregroundStyle(.secondary)
                Text("Model has no sampling recommendations")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }

        // Enable sampling toggle (only when not using recommended or model has no recommendations)
        if !useRecommendedSampling || !hasRecommendedSampling {
            Toggle("Enable Sampling", isOn: $doSample)
                .onChange(of: doSample) { _, newValue in
                    if !newValue {
                        // Switching to greedy - set temperature to 0
                        temperature = 0.0
                    } else if temperature == 0 {
                        // Switching to sampling - set reasonable default
                        temperature = 0.7
                    }
                }
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
                Text("\(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.0.0") (\(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "1"))")
                    .foregroundStyle(.secondary)
            }

            HStack {
                Text("Device")
                Spacer()
                Text(DeviceType.chipName)
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

            Button {
                showingAcknowledgements = true
            } label: {
                HStack {
                    Label("Acknowledgements", systemImage: "doc.text")
                    Spacer()
                    Image(systemName: "chevron.right")
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
        // Clamp maxTokens to current model's context size
        maxTokens = min(chatVM.maxTokens, maxTokensLimit)

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

        // Load sampling settings from InferenceService (which already loaded from storage)
        doSample = InferenceService.shared.doSample
        topP = InferenceService.shared.topP
        topK = InferenceService.shared.topK
        useRecommendedSampling = InferenceService.shared.useRecommendedSampling

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
            // Save sampling settings
            await StorageService.shared.saveDoSample(doSample)
            await StorageService.shared.saveTopP(topP)
            await StorageService.shared.saveTopK(topK)
            await StorageService.shared.saveUseRecommendedSampling(useRecommendedSampling)
            // Update InferenceService settings
            await MainActor.run {
                InferenceService.shared.debugLevel = debugLevel
                InferenceService.shared.repetitionDetectionEnabled = repetitionDetectionEnabled
                InferenceService.shared.doSample = doSample
                InferenceService.shared.topP = topP
                InferenceService.shared.topK = topK
                InferenceService.shared.useRecommendedSampling = useRecommendedSampling
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
        doSample = StorageService.defaultDoSampleValue
        topP = StorageService.defaultTopPValue
        topK = StorageService.defaultTopKValue
        useRecommendedSampling = StorageService.defaultUseRecommendedSamplingValue

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
                InferenceService.shared.doSample = doSample
                InferenceService.shared.topP = topP
                InferenceService.shared.topK = topK
                InferenceService.shared.useRecommendedSampling = useRecommendedSampling
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

// MARK: - Acknowledgements View

struct OpenSourceLibrary: Identifiable {
    let id = UUID()
    let name: String
    let url: String
    let license: String
    let licenseType: String
    let copyright: String
}

private let openSourceLibraries: [OpenSourceLibrary] = [
    OpenSourceLibrary(
        name: "swift-transformers",
        url: "https://github.com/huggingface/swift-transformers",
        license: """
        Apache License, Version 2.0

        Copyright 2022 Hugging Face SAS.

        Licensed under the Apache License, Version 2.0 (the "License"); \
        you may not use this file except in compliance with the License. \
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software \
        distributed under the License is distributed on an "AS IS" BASIS, \
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. \
        See the License for the specific language governing permissions and \
        limitations under the License.
        """,
        licenseType: "Apache 2.0",
        copyright: "2022 Hugging Face SAS"
    ),
    OpenSourceLibrary(
        name: "Yams",
        url: "https://github.com/jpsim/Yams",
        license: """
        The MIT License (MIT)

        Copyright (c) 2016 JP Simard.

        Permission is hereby granted, free of charge, to any person obtaining a copy \
        of this software and associated documentation files (the "Software"), to deal \
        in the Software without restriction, including without limitation the rights \
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell \
        copies of the Software, and to permit persons to whom the Software is \
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all \
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR \
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, \
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE \
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER \
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, \
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE \
        SOFTWARE.
        """,
        licenseType: "MIT",
        copyright: "2016 JP Simard"
    ),
    OpenSourceLibrary(
        name: "Stencil",
        url: "https://github.com/stencilproject/Stencil",
        license: """
        BSD 2-Clause License

        Copyright (c) 2022, Kyle Fuller
        All rights reserved.

        Redistribution and use in source and binary forms, with or without \
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this \
        list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright notice, \
        this list of conditions and the following disclaimer in the documentation \
        and/or other materials provided with the distribution.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" \
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE \
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE \
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE \
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL \
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR \
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER \
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, \
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE \
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """,
        licenseType: "BSD 2-Clause",
        copyright: "2022 Kyle Fuller"
    ),
    OpenSourceLibrary(
        name: "swift-argument-parser",
        url: "https://github.com/apple/swift-argument-parser",
        license: """
        Apache License, Version 2.0 with Runtime Library Exception

        Copyright (c) Apple Inc.

        Licensed under the Apache License, Version 2.0 (the "License"); \
        you may not use this file except in compliance with the License. \
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software \
        distributed under the License is distributed on an "AS IS" BASIS, \
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. \
        See the License for the specific language governing permissions and \
        limitations under the License.

        Runtime Library Exception: As an exception, if you use this Software to \
        compile your source code and portions of this Software are embedded into \
        the binary product as a result, you may redistribute such product without \
        providing attribution as would otherwise be required by Sections 4(a), \
        4(b) and 4(d) of the License.
        """,
        licenseType: "Apache 2.0",
        copyright: "Apple Inc."
    ),
    OpenSourceLibrary(
        name: "swift-collections",
        url: "https://github.com/apple/swift-collections",
        license: """
        Apache License, Version 2.0 with Runtime Library Exception

        Copyright (c) Apple Inc.

        Licensed under the Apache License, Version 2.0 (the "License"); \
        you may not use this file except in compliance with the License. \
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software \
        distributed under the License is distributed on an "AS IS" BASIS, \
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. \
        See the License for the specific language governing permissions and \
        limitations under the License.

        Runtime Library Exception: As an exception, if you use this Software to \
        compile your source code and portions of this Software are embedded into \
        the binary product as a result, you may redistribute such product without \
        providing attribution as would otherwise be required by Sections 4(a), \
        4(b) and 4(d) of the License.
        """,
        licenseType: "Apache 2.0",
        copyright: "Apple Inc."
    ),
    OpenSourceLibrary(
        name: "Jinja",
        url: "https://github.com/maiqingqiang/Jinja",
        license: """
        MIT License

        Copyright (c) 2024 John Mai

        Permission is hereby granted, free of charge, to any person obtaining a copy \
        of this software and associated documentation files (the "Software"), to deal \
        in the Software without restriction, including without limitation the rights \
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell \
        copies of the Software, and to permit persons to whom the Software is \
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all \
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR \
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, \
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE \
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER \
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, \
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE \
        SOFTWARE.
        """,
        licenseType: "MIT",
        copyright: "2024 John Mai"
    ),
    OpenSourceLibrary(
        name: "PathKit",
        url: "https://github.com/kylef/PathKit",
        license: """
        BSD 2-Clause License

        Copyright (c) 2014, Kyle Fuller
        All rights reserved.

        Redistribution and use in source and binary forms, with or without \
        modification, are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this \
        list of conditions and the following disclaimer.
        2. Redistributions in binary form must reproduce the above copyright notice, \
        this list of conditions and the following disclaimer in the documentation \
        and/or other materials provided with the distribution.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND \
        ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED \
        WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE \
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR \
        ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES \
        (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; \
        LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND \
        ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT \
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS \
        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """,
        licenseType: "BSD 2-Clause",
        copyright: "2014 Kyle Fuller"
    ),
    OpenSourceLibrary(
        name: "Spectre",
        url: "https://github.com/kylef/Spectre",
        license: """
        BSD 2-Clause License

        Copyright (c) 2015, Kyle Fuller
        All rights reserved.

        Redistribution and use in source and binary forms, with or without \
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this \
        list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright notice, \
        this list of conditions and the following disclaimer in the documentation \
        and/or other materials provided with the distribution.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" \
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE \
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE \
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE \
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL \
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR \
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER \
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, \
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE \
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """,
        licenseType: "BSD 2-Clause",
        copyright: "2015 Kyle Fuller"
    ),
]

struct AcknowledgementsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var selectedLibrary: OpenSourceLibrary?

    var body: some View {
        NavigationStack {
            List(openSourceLibraries) { library in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(library.name)
                            .font(.body)
                            .foregroundStyle(.primary)
                        Text(library.licenseType)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    selectedLibrary = library
                }
            }
            .navigationTitle("Acknowledgements")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") {
                        dismiss()
                    }
                    .keyboardShortcut(.escape, modifiers: [])
                }
            }
            .sheet(item: $selectedLibrary) { library in
                LicenseDetailView(library: library, onDismiss: { selectedLibrary = nil })
            }
        }
        #if os(macOS)
        .frame(minWidth: 400, minHeight: 350)
        #endif
    }
}

struct LicenseDetailView: View {
    let library: OpenSourceLibrary
    let onDismiss: () -> Void

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Link(destination: URL(string: library.url)!) {
                        HStack {
                            Text(library.url)
                                .font(.caption)
                            Image(systemName: "arrow.up.right")
                                .font(.caption2)
                        }
                        .foregroundStyle(.blue)
                    }

                    Divider()

                    Text(library.license)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                .padding()
            }
            .navigationTitle(library.name)
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") {
                        onDismiss()
                    }
                    .keyboardShortcut(.escape, modifiers: [])
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 450, minHeight: 350)
        #endif
    }
}

#Preview {
    NavigationStack {
        SettingsView()
            .environment(ChatViewModel())
    }
}
