//
//  SettingsView.swift
//  ANEMLLChat
//
//  App settings and configuration
//

import SwiftUI

// System prompt options
enum SystemPromptOption: String, CaseIterable, Identifiable {
    case modelDefault = "Model's Default"
    case modelThinking = "Model's Default (Thinking)"
    case modelNonThinking = "Model's Default (Non-Thinking)"
    case noPrompt = "No Prompt"
    case custom = "Custom"

    var id: String { rawValue }
}

struct SettingsView: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(\.dismiss) private var dismiss

    @State private var temperature: Float = 0.0  // Default: greedy decoding
    @State private var maxTokens: Int = 512
    @State private var systemPromptOption: SystemPromptOption = .modelDefault
    @State private var customPrompt: String = ""

    @State private var showingLogs = false
    @State private var autoLoadLastModel = true
    @State private var debugLevel: Int = 0

    var body: some View {
        Form {
            // Model settings
            modelSection

            // Generation settings
            generationSection

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
        .sheet(isPresented: $showingLogs) {
            LogsView()
        }
    }

    // MARK: - Model Section

    private var modelSection: some View {
        Section {
            Toggle("Auto-load last model", isOn: $autoLoadLastModel)

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
            Text("When enabled, the app will automatically load the last used model on startup")
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
        } header: {
            Text("Generation")
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
            case .modelDefault:
                Text("Uses the model's built-in default prompt")
            case .modelThinking:
                Text("Uses thinking/reasoning mode if supported")
            case .modelNonThinking:
                Text("Uses non-thinking mode if supported")
            case .noPrompt:
                Text("No system prompt - raw model output")
            case .custom:
                Text("Custom instructions for the AI")
            }
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
        if storedPrompt.isEmpty {
            systemPromptOption = .noPrompt
        } else if storedPrompt.hasPrefix("[MODEL_DEFAULT]") {
            systemPromptOption = .modelDefault
        } else if storedPrompt.hasPrefix("[MODEL_THINKING]") {
            systemPromptOption = .modelThinking
        } else if storedPrompt.hasPrefix("[MODEL_NON_THINKING]") {
            systemPromptOption = .modelNonThinking
        } else {
            systemPromptOption = .custom
            customPrompt = storedPrompt
        }

        Task {
            autoLoadLastModel = await StorageService.shared.autoLoadLastModel
            debugLevel = await StorageService.shared.debugLevel
        }
    }

    private func saveSettings() {
        chatVM.temperature = temperature
        chatVM.maxTokens = maxTokens

        // Convert option to stored string
        switch systemPromptOption {
        case .modelDefault:
            chatVM.systemPrompt = "[MODEL_DEFAULT]"
        case .modelThinking:
            chatVM.systemPrompt = "[MODEL_THINKING]"
        case .modelNonThinking:
            chatVM.systemPrompt = "[MODEL_NON_THINKING]"
        case .noPrompt:
            chatVM.systemPrompt = ""
        case .custom:
            chatVM.systemPrompt = customPrompt
        }

        Task {
            await chatVM.saveSettings()
            await StorageService.shared.saveAutoLoadLastModel(autoLoadLastModel)
            await StorageService.shared.saveDebugLevel(debugLevel)
            // Update InferenceService debug level
            await MainActor.run {
                InferenceService.shared.debugLevel = debugLevel
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
