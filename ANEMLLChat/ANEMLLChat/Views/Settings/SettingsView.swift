//
//  SettingsView.swift
//  ANEMLLChat
//
//  App settings and configuration
//

import SwiftUI

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
    @Environment(\.dismiss) private var dismiss

    @State private var temperature: Float = 0.0
    @State private var maxTokens: Int = 2048
    @State private var systemPromptOption: SystemPromptOption = .defaultPrompt
    @State private var customPrompt: String = ""

    @State private var showingLogs = false
    @State private var autoLoadLastModel = true
    @State private var debugLevel: Int = 0
    @State private var repetitionDetectionEnabled = false
    @State private var debugDisablePrefill = StorageService.defaultDebugDisablePrefillValue
    @State private var debugContextCap = StorageService.defaultDebugContextCapValue
    @State private var debugDisableIOBackings = StorageService.defaultDebugDisableIOBackingsValue
    @State private var debugRepeatInferCount = StorageService.defaultDebugRepeatInferCountValue
    @State private var debugRepeatOnlyDivergence = StorageService.defaultDebugRepeatOnlyDivergenceValue
    @State private var enableMarkup = StorageService.defaultEnableMarkupValue
    @State private var sendButtonOnLeft = StorageService.defaultSendButtonOnLeftValue
    @State private var loadLastChat = StorageService.defaultLoadLastChatValue
    @State private var largeControls = StorageService.defaultLargeControlsValue
    @State private var showMicrophone = StorageService.defaultShowMicrophoneValue
    @State private var showingResetConfirmation = false

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

            // Debug inference (debug builds only)
            if DebugInferenceOptions.isEnabled {
                debugInferenceSection
            }

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
        } header: {
            Text("Model")
        } footer: {
            Text(loadLastChat ? "App will restore your last conversation on startup" : "App will start with a new chat on startup")
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

    // MARK: - Debug Inference Section

    private var debugInferenceSection: some View {
        Section {
            Toggle("Disable Prefill (Inference Only)", isOn: $debugDisablePrefill)

            Toggle("Disable CoreML I/O Backings (CVPixelBuffer)", isOn: $debugDisableIOBackings)

            Stepper(value: $debugContextCap, in: 0...512, step: 16) {
                HStack {
                    Text("Context Cap")
                    Spacer()
                    Text(debugContextCap == 0 ? "Off" : "\(debugContextCap)")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }

            Stepper(value: $debugRepeatInferCount, in: 0...4, step: 1) {
                HStack {
                    Text("Repeat Infer (2-4)")
                    Spacer()
                    Text(debugRepeatInferCount < 2 ? "Off" : "\(debugRepeatInferCount)x")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }

            Toggle("Only Log Divergence", isOn: $debugRepeatOnlyDivergence)
        } header: {
            Text("Debug Inference")
        } footer: {
            Text("Disables prefill, CoreML I/O backings, caps the input context, and optionally repeats token inference for divergence testing (heavy). Requires model reload. Debug builds only.")
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
            debugDisablePrefill = await StorageService.shared.debugDisablePrefill
            debugContextCap = await StorageService.shared.debugContextCap
            debugDisableIOBackings = await StorageService.shared.debugDisableIOBackings
            debugRepeatInferCount = await StorageService.shared.debugRepeatInferCount
            debugRepeatOnlyDivergence = await StorageService.shared.debugRepeatOnlyDivergence
            enableMarkup = await StorageService.shared.enableMarkup
            sendButtonOnLeft = await StorageService.shared.sendButtonOnLeft
            loadLastChat = await StorageService.shared.loadLastChat
            largeControls = await StorageService.shared.largeControls
            showMicrophone = await StorageService.shared.showMicrophone
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
            await StorageService.shared.saveDebugDisablePrefill(debugDisablePrefill)
            await StorageService.shared.saveDebugContextCap(debugContextCap)
            await StorageService.shared.saveDebugDisableIOBackings(debugDisableIOBackings)
            await StorageService.shared.saveDebugRepeatInferCount(debugRepeatInferCount)
            await StorageService.shared.saveDebugRepeatOnlyDivergence(debugRepeatOnlyDivergence)
            await StorageService.shared.saveEnableMarkup(enableMarkup)
            await StorageService.shared.saveSendButtonOnLeft(sendButtonOnLeft)
            await StorageService.shared.saveLoadLastChat(loadLastChat)
            await StorageService.shared.saveLargeControls(largeControls)
            await StorageService.shared.saveShowMicrophone(showMicrophone)
            // Update InferenceService settings
            await MainActor.run {
                InferenceService.shared.debugLevel = debugLevel
                InferenceService.shared.repetitionDetectionEnabled = repetitionDetectionEnabled
                if DebugInferenceOptions.isEnabled {
                    InferenceService.shared.debugDisablePrefill = debugDisablePrefill
                    InferenceService.shared.debugContextCap = debugContextCap
                    InferenceService.shared.debugDisableIOBackings = debugDisableIOBackings
                    InferenceService.shared.debugRepeatInferCount = debugRepeatInferCount
                    InferenceService.shared.debugRepeatOnlyDivergence = debugRepeatOnlyDivergence
                } else {
                    InferenceService.shared.debugDisablePrefill = false
                    InferenceService.shared.debugContextCap = 0
                    InferenceService.shared.debugDisableIOBackings = false
                    InferenceService.shared.debugRepeatInferCount = 0
                    InferenceService.shared.debugRepeatOnlyDivergence = false
                }
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
        debugDisablePrefill = StorageService.defaultDebugDisablePrefillValue
        debugContextCap = StorageService.defaultDebugContextCapValue
        debugDisableIOBackings = StorageService.defaultDebugDisableIOBackingsValue
        debugRepeatInferCount = StorageService.defaultDebugRepeatInferCountValue
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
                if DebugInferenceOptions.isEnabled {
                    InferenceService.shared.debugDisablePrefill = debugDisablePrefill
                    InferenceService.shared.debugContextCap = debugContextCap
                    InferenceService.shared.debugDisableIOBackings = debugDisableIOBackings
                    InferenceService.shared.debugRepeatInferCount = debugRepeatInferCount
                } else {
                    InferenceService.shared.debugDisablePrefill = false
                    InferenceService.shared.debugContextCap = 0
                    InferenceService.shared.debugDisableIOBackings = false
                    InferenceService.shared.debugRepeatInferCount = 0
                }
            }
        }
    }
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
