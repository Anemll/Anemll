//
//  SettingsView.swift
//  ANEMLLChat
//
//  App settings and configuration
//

import SwiftUI
import UniformTypeIdentifiers
#if os(macOS)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

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
    #if os(macOS) || os(tvOS)
    @Environment(ModelManagerViewModel.self) private var modelManager
    @ObservedObject private var openAIServer = OpenAICompatibleServerService.shared
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
    #if os(macOS) || os(tvOS)
    @State private var openAICompatibleServerEnabled = StorageService.defaultOpenAICompatibleServerEnabledValue
    @State private var openAICompatibleServerBindMode = StorageService.defaultOpenAICompatibleServerBindModeValue
    @State private var openAICompatibleServerPort = StorageService.defaultOpenAICompatibleServerPortValue
    @State private var openAICompatibleServerStatusMessage: String?
    @State private var hasLoadedOpenAICompatibleServerSettings = false
    #endif

    var body: some View {
        Group {
            #if os(tvOS)
            tvSettingsBody
            #else
            Form {
                // Model settings
                modelSection

                // Generation settings
                generationSection

                // Display settings
                displaySection

                // System prompt
                systemPromptSection

                #if os(macOS)
                // Server setup
                openAICompatibleServerSection
                #endif

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
            #endif
        }
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

    // MARK: - tvOS Settings

    #if os(tvOS)
    private var tvSettingsBody: some View {
        VStack(spacing: 0) {
            // Header bar
            HStack(alignment: .center) {
                Button {
                    saveSettings()
                    dismiss()
                } label: {
                    Text("Done")
                }
                .buttonStyle(.bordered)

                Spacer()

                Text("Settings")
                    .font(.title2)
                    .fontWeight(.bold)

                Spacer()

                // Invisible spacer to balance Done button
                Text("Done").opacity(0)
            }
            .padding(.horizontal, 48)
            .padding(.top, 24)
            .padding(.bottom, 16)
            .focusSection()

            ScrollView {
                VStack(spacing: 24) {
                    tvModelCard.focusSection()
                    tvGenerationCard.focusSection()
                    tvDisplayCard.focusSection()
                    tvSystemPromptCard.focusSection()
                    tvServerCard.focusSection()
                    tvDebugCard.focusSection()
                    tvAboutCard.focusSection()
                }
                .padding(.horizontal, 48)
                .padding(.top, 8)
                .padding(.bottom, 40)
            }
        }
    }

    private func tvSectionHeader(_ title: String, icon: String, color: Color) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(color)
            Text(title)
                .font(.title3)
                .fontWeight(.semibold)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func tvCardBackground() -> some View {
        RoundedRectangle(cornerRadius: 16, style: .continuous)
            .fill(Color.white.opacity(0.06))
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
            )
    }

    // MARK: - tvOS Model Card

    private var tvModelCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            tvSectionHeader("Model", icon: "cpu", color: .blue)

            Toggle("Auto-load last model", isOn: $autoLoadLastModel)
            Toggle("Load last chat on startup", isOn: $loadLastChat)

            Button(role: .destructive) {
                Task {
                    await StorageService.shared.clearLastModel()
                }
            } label: {
                Label("Clear remembered model", systemImage: "xmark.circle")
            }
            .buttonStyle(.bordered)
        }
        .padding(24)
        .background(tvCardBackground())
    }

    // MARK: - tvOS Generation Card

    private var tvGenerationCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            tvSectionHeader("Generation", icon: "slider.horizontal.3", color: .orange)

            // Sampling controls
            samplingControls

            // Temperature
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Temperature")
                    Spacer()
                    Text(String(format: "%.2f", temperature))
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }

                tvAdjustmentButtons(
                    label: "Adjust Temperature",
                    decrementAction: {
                        temperature = max(0, temperature - 0.05)
                    },
                    incrementAction: {
                        temperature = min(2, temperature + 0.05)
                    },
                    disabled: useRecommendedSampling && hasRecommendedSampling
                )

                Text("Lower = more focused, Higher = more creative")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Top-P and Top-K (only when sampling is enabled)
            if doSample || (useRecommendedSampling && hasRecommendedSampling) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Top-P")
                        Spacer()
                        Text(String(format: "%.2f", topP))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }

                    tvAdjustmentButtons(
                        label: "Adjust Top-P",
                        decrementAction: {
                            topP = max(0, topP - 0.05)
                        },
                        incrementAction: {
                            topP = min(1, topP + 0.05)
                        },
                        disabled: useRecommendedSampling && hasRecommendedSampling
                    )

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

                    tvAdjustmentButtons(
                        label: "Adjust Top-K",
                        decrementAction: {
                            topK = max(0, topK - 5)
                        },
                        incrementAction: {
                            topK = min(100, topK + 5)
                        },
                        disabled: useRecommendedSampling && hasRecommendedSampling
                    )

                    Text("Top-K sampling (0 = disabled)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Max Tokens
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Max Tokens")
                    Spacer()
                    Text("\(maxTokens)")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }

                tvAdjustmentButtons(
                    label: "Adjust Max Tokens",
                    decrementAction: {
                        maxTokens = max(64, maxTokens - 64)
                    },
                    incrementAction: {
                        maxTokens = min(maxTokensLimit, maxTokens + 64)
                    }
                )

                Text("Maximum number of tokens to generate (model max: \(maxTokensLimit))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Toggle("Repetition Detection", isOn: $repetitionDetectionEnabled)

            Text(repetitionDetectionEnabled ? "Stops generation if repetitive patterns are detected" : "Generation continues until EOS or max tokens (CLI behavior)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(24)
        .background(tvCardBackground())
    }

    // MARK: - tvOS Display Card

    private var tvDisplayCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            tvSectionHeader("Display", icon: "paintbrush", color: .purple)

            Toggle("Enable Markup", isOn: $enableMarkup)
            Toggle("Send Button on Left", isOn: $sendButtonOnLeft)

            Text("tvOS uses focus-driven controls optimized for Apple TV remote navigation")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(24)
        .background(tvCardBackground())
    }

    // MARK: - tvOS System Prompt Card

    private var tvSystemPromptCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            tvSectionHeader("System Prompt", icon: "text.quote", color: .cyan)

            ForEach(SystemPromptOption.allCases) { option in
                TVPromptOptionButton(
                    option: option,
                    isSelected: systemPromptOption == option,
                    description: tvPromptDescription(for: option)
                ) {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        systemPromptOption = option
                    }
                }
            }

            if systemPromptOption == .custom {
                TextField("Custom system prompt", text: $customPrompt)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(Color.white.opacity(0.08))
                    )
            }
        }
        .padding(24)
        .background(tvCardBackground())
    }

    private func tvPromptDescription(for option: SystemPromptOption) -> String {
        switch option {
        case .defaultPrompt:
            return "Standard inference with chat template"
        case .noTemplate:
            return "Raw inference without chat template"
        case .modelThinking:
            return "Thinking/reasoning mode if supported"
        case .modelNonThinking:
            return "Non-thinking mode if supported"
        case .custom:
            return "Custom system prompt instructions"
        }
    }

    // MARK: - tvOS Server Card

    private var tvServerCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            tvSectionHeader("OpenAI API Server", icon: "network", color: .green)

            Toggle("Enable OpenAI-compatible Server", isOn: $openAICompatibleServerEnabled)

            Picker("Network", selection: $openAICompatibleServerBindMode) {
                ForEach(OpenAICompatibleServerBindMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .disabled(!openAICompatibleServerEnabled)

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Port")
                    Spacer()
                    Text("\(openAICompatibleServerPort)")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }

                tvAdjustmentButtons(
                    label: "Adjust Port",
                    decrementAction: {
                        openAICompatibleServerPort = max(1, openAICompatibleServerPort - 1)
                    },
                    incrementAction: {
                        openAICompatibleServerPort = min(65_535, openAICompatibleServerPort + 1)
                    },
                    disabled: !openAICompatibleServerEnabled
                )
            }

            if openAICompatibleServerEnabled {
                HStack(spacing: 6) {
                    Image(systemName: openAIServer.isRunning ? "checkmark.circle.fill" : "clock.arrow.circlepath")
                        .foregroundStyle(openAIServer.isRunning ? .green : .orange)
                    Text(openAIServer.isRunning ? "Server is running" : "Starting server...")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }

                if let localhostURL = openAIServer.localhostURL {
                    HStack(spacing: 8) {
                        Text("Localhost")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .frame(width: 90, alignment: .leading)
                        Text(localhostURL)
                            .font(.callout)
                            .lineLimit(2)
                    }
                }

                if openAICompatibleServerBindMode == .lan {
                    if let lanURL = openAIServer.lanURL {
                        HStack(spacing: 8) {
                            Text("LAN/WiFi")
                                .font(.callout)
                                .foregroundStyle(.secondary)
                                .frame(width: 90, alignment: .leading)
                            Text(lanURL)
                                .font(.callout)
                                .lineLimit(2)
                        }
                    } else {
                        Text("No active LAN/WiFi IPv4 interface found.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if let message = openAICompatibleServerStatusMessage {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text("Exposes /v1/models and /v1/chat/completions for the currently loaded model.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(24)
        .background(tvCardBackground())
        .onChange(of: openAICompatibleServerEnabled) { _, _ in
            applyOpenAICompatibleServerSettings()
        }
        .onChange(of: openAICompatibleServerBindMode) { _, _ in
            applyOpenAICompatibleServerSettings()
        }
        .onChange(of: openAICompatibleServerPort) { _, newValue in
            let normalized = min(max(newValue, 1), 65_535)
            if normalized != newValue {
                openAICompatibleServerPort = normalized
                return
            }
            applyOpenAICompatibleServerSettings()
        }
    }

    // MARK: - tvOS Debug Card

    private var tvDebugCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            tvSectionHeader("Debug", icon: "ladybug", color: .yellow)

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
            .buttonStyle(.bordered)

            Text("Debug level affects console output during model loading and inference")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(24)
        .background(tvCardBackground())
    }

    // MARK: - tvOS About Card

    private var tvAboutCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            tvSectionHeader("About", icon: "info.circle", color: .gray)

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
            .buttonStyle(.bordered)

            Link(destination: URL(string: "https://huggingface.co/anemll")!) {
                HStack {
                    Label("HuggingFace Models", systemImage: "link")
                    Spacer()
                    Image(systemName: "arrow.up.right")
                        .foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.bordered)

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
            .buttonStyle(.bordered)

            Button(role: .destructive) {
                showingResetConfirmation = true
            } label: {
                Label("Reset to Defaults", systemImage: "arrow.counterclockwise")
            }
            .buttonStyle(.bordered)
            .confirmationDialog("Reset all settings to defaults?", isPresented: $showingResetConfirmation, titleVisibility: .visible) {
                Button("Reset", role: .destructive) {
                    resetToDefaults()
                }
                Button("Cancel", role: .cancel) {}
            }

            Text("ANEMLL Chat - On-device LLM inference powered by Apple Neural Engine")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(24)
        .background(tvCardBackground())
    }
    #endif

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

    #if os(macOS) || os(tvOS)
    private var openAICompatibleServerSection: some View {
        Section {
            Toggle("Enable OpenAI-compatible Server", isOn: $openAICompatibleServerEnabled)

            Picker("Network", selection: $openAICompatibleServerBindMode) {
                ForEach(OpenAICompatibleServerBindMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .disabled(!openAICompatibleServerEnabled)

            #if os(tvOS)
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Port")
                    Spacer()
                    Text("\(openAICompatibleServerPort)")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }

                tvAdjustmentButtons(
                    label: "Adjust Port",
                    decrementAction: {
                        openAICompatibleServerPort = max(1, openAICompatibleServerPort - 1)
                    },
                    incrementAction: {
                        openAICompatibleServerPort = min(65_535, openAICompatibleServerPort + 1)
                    },
                    disabled: !openAICompatibleServerEnabled
                )
            }
            #else
            HStack(spacing: 8) {
                Text("Port")
                Spacer()
                TextField(
                    "Port",
                    value: $openAICompatibleServerPort,
                    format: .number
                )
                .frame(width: 90)
                .multilineTextAlignment(.trailing)
                .disabled(!openAICompatibleServerEnabled)

                Stepper(
                    "",
                    value: $openAICompatibleServerPort,
                    in: 1...65_535
                )
                .labelsHidden()
                .disabled(!openAICompatibleServerEnabled)
            }
            #endif

            if openAICompatibleServerEnabled {
                HStack(spacing: 6) {
                    Image(systemName: openAIServer.isRunning ? "checkmark.circle.fill" : "clock.arrow.circlepath")
                        .foregroundStyle(openAIServer.isRunning ? .green : .orange)
                    Text(openAIServer.isRunning ? "Server is running" : "Starting server...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let localhostURL = openAIServer.localhostURL {
                    openAIURLRow(title: "Localhost", url: localhostURL)
                }

                if openAICompatibleServerBindMode == .lan {
                    if let lanURL = openAIServer.lanURL {
                        openAIURLRow(title: "LAN/WiFi", url: lanURL)
                    } else {
                        Text("No active LAN/WiFi IPv4 interface found.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if let message = openAICompatibleServerStatusMessage {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } header: {
            Text("OpenAI API Server")
        } footer: {
            Text("Exposes `/v1/models` and `/v1/chat/completions` for the currently loaded model.")
        }
        .onChange(of: openAICompatibleServerEnabled) { _, _ in
            applyOpenAICompatibleServerSettings()
        }
        .onChange(of: openAICompatibleServerBindMode) { _, _ in
            applyOpenAICompatibleServerSettings()
        }
        .onChange(of: openAICompatibleServerPort) { _, newValue in
            let normalized = min(max(newValue, 1), 65_535)
            if normalized != newValue {
                openAICompatibleServerPort = normalized
                return
            }
            applyOpenAICompatibleServerSettings()
        }
    }

    private func openAIURLRow(title: String, url: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 72, alignment: .leading)

            Text(url)
                .font(.caption)
                #if !os(tvOS)
                .textSelection(.enabled)
                #endif
                .lineLimit(2)

            Spacer(minLength: 8)

            #if os(macOS)
            Button("Copy") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(url, forType: .string)
                openAICompatibleServerStatusMessage = "\(title) URL copied to clipboard."
            }
            .buttonStyle(.borderless)
            #elseif os(tvOS)
            // tvOS: no clipboard, just display the URL
            #endif
        }
    }
    #endif

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

                #if os(tvOS)
                tvAdjustmentButtons(
                    label: "Adjust Temperature",
                    decrementAction: {
                        temperature = max(0, temperature - 0.05)
                    },
                    incrementAction: {
                        temperature = min(2, temperature + 0.05)
                    },
                    disabled: useRecommendedSampling && hasRecommendedSampling
                )
                #else
                Slider(value: $temperature, in: 0...2, step: 0.05)
                    .disabled(useRecommendedSampling && hasRecommendedSampling)
                #endif

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

                    #if os(tvOS)
                    tvAdjustmentButtons(
                        label: "Adjust Top-P",
                        decrementAction: {
                            topP = max(0, topP - 0.05)
                        },
                        incrementAction: {
                            topP = min(1, topP + 0.05)
                        },
                        disabled: useRecommendedSampling && hasRecommendedSampling
                    )
                    #else
                    Slider(value: $topP, in: 0...1, step: 0.05)
                        .disabled(useRecommendedSampling && hasRecommendedSampling)
                    #endif

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

                    #if os(tvOS)
                    tvAdjustmentButtons(
                        label: "Adjust Top-K",
                        decrementAction: {
                            topK = max(0, topK - 5)
                        },
                        incrementAction: {
                            topK = min(100, topK + 5)
                        },
                        disabled: useRecommendedSampling && hasRecommendedSampling
                    )
                    #else
                    Slider(
                        value: Binding(
                            get: { Double(topK) },
                            set: { topK = Int($0) }
                        ),
                        in: 0...100,
                        step: 5
                    )
                    .disabled(useRecommendedSampling && hasRecommendedSampling)
                    #endif

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

                #if os(tvOS)
                tvAdjustmentButtons(
                    label: "Adjust Max Tokens",
                    decrementAction: {
                        maxTokens = max(64, maxTokens - 64)
                    },
                    incrementAction: {
                        maxTokens = min(maxTokensLimit, maxTokens + 64)
                    }
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(maxTokens) },
                        set: { maxTokens = Int($0) }
                    ),
                    in: 64...Double(maxTokensLimit),
                    step: 64
                )
                #endif

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

    #if os(tvOS)
    private func tvAdjustmentButtons(
        label: String,
        decrementAction: @escaping () -> Void,
        incrementAction: @escaping () -> Void,
        disabled: Bool = false
    ) -> some View {
        HStack(spacing: 12) {
            Button(action: decrementAction) {
                Image(systemName: "minus.circle.fill")
            }
            .buttonStyle(.bordered)
            .disabled(disabled)

            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)

            Button(action: incrementAction) {
                Image(systemName: "plus.circle.fill")
            }
            .buttonStyle(.bordered)
            .disabled(disabled)
        }
    }
    #endif

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
                #if os(tvOS)
                TextField("Custom system prompt", text: $customPrompt)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(Color.white.opacity(0.08))
                    )
                #else
                TextEditor(text: $customPrompt)
                    .frame(minHeight: 80)
                    .font(.body)
                #endif
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
            #if !os(tvOS)
            Toggle("Show Microphone", isOn: $showMicrophone)
            #endif
            #if os(iOS) || os(visionOS)
            Toggle("Large Controls", isOn: $largeControls)
            #endif
        } header: {
            Text("Display")
        } footer: {
            #if os(iOS) || os(visionOS)
            Text(largeControls ? "Send button and toolbar icons are enlarged for easier touch" : "Standard control sizes")
            #elseif os(tvOS)
            Text("tvOS uses focus-driven controls optimized for Apple TV remote navigation")
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
            #if os(macOS) || os(tvOS)
            openAICompatibleServerEnabled = await StorageService.shared.openAICompatibleServerEnabled
            openAICompatibleServerBindMode = await StorageService.shared.openAICompatibleServerBindMode
            openAICompatibleServerPort = await StorageService.shared.openAICompatibleServerPort
            hasLoadedOpenAICompatibleServerSettings = true
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
            #if os(macOS) || os(tvOS)
            await openAIServer.applySettings(
                enabled: openAICompatibleServerEnabled,
                bindMode: openAICompatibleServerBindMode,
                port: openAICompatibleServerPort
            )
            #endif
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
        #if os(macOS) || os(tvOS)
        openAICompatibleServerEnabled = StorageService.defaultOpenAICompatibleServerEnabledValue
        openAICompatibleServerBindMode = StorageService.defaultOpenAICompatibleServerBindModeValue
        openAICompatibleServerPort = StorageService.defaultOpenAICompatibleServerPortValue
        #endif

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
            #if os(macOS) || os(tvOS)
            await openAIServer.applySettings(
                enabled: openAICompatibleServerEnabled,
                bindMode: openAICompatibleServerBindMode,
                port: openAICompatibleServerPort
            )
            #endif
        }
    }

    #if os(macOS) || os(tvOS)
    private func applyOpenAICompatibleServerSettings() {
        guard hasLoadedOpenAICompatibleServerSettings else { return }

        let normalizedPort = min(max(openAICompatibleServerPort, 1), 65_535)
        if openAICompatibleServerPort != normalizedPort {
            openAICompatibleServerPort = normalizedPort
            return
        }

        Task {
            await openAIServer.applySettings(
                enabled: openAICompatibleServerEnabled,
                bindMode: openAICompatibleServerBindMode,
                port: normalizedPort
            )

            await MainActor.run {
                if let error = openAIServer.lastErrorMessage {
                    openAICompatibleServerStatusMessage = error
                } else if openAICompatibleServerEnabled {
                    openAICompatibleServerStatusMessage = openAIServer.isRunning
                        ? "Server running on port \(normalizedPort)."
                        : "Starting server on port \(normalizedPort)..."
                } else {
                    openAICompatibleServerStatusMessage = "Server disabled."
                }
            }
        }
    }
    #endif

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

// MARK: - tvOS Prompt Option Button

#if os(tvOS)
private struct TVPromptOptionButton: View {
    let option: SystemPromptOption
    let isSelected: Bool
    let description: String
    let action: () -> Void

    @FocusState private var isFocused: Bool

    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                // Selection indicator
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.title3)
                    .foregroundStyle(isSelected ? .cyan : .secondary)

                VStack(alignment: .leading, spacing: 3) {
                    Text(option.rawValue)
                        .font(.callout)
                        .fontWeight(isSelected ? .semibold : .regular)
                        .foregroundStyle(isSelected ? .primary : .secondary)

                    Text(description)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }

                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(isSelected
                          ? Color.cyan.opacity(isFocused ? 0.18 : 0.10)
                          : Color.white.opacity(isFocused ? 0.10 : 0.04))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(
                        isFocused
                            ? Color.cyan.opacity(0.6)
                            : (isSelected ? Color.cyan.opacity(0.25) : Color.clear),
                        lineWidth: 1.5
                    )
            )
            .scaleEffect(isFocused ? 1.02 : 1.0)
            .animation(.easeOut(duration: 0.15), value: isFocused)
        }
        .buttonStyle(.plain)
        .focusable()
        .focused($isFocused)
    }
}
#endif

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
                            .selectable(true)
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
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #elseif os(iOS) || os(visionOS)
        UIPasteboard.general.string = text
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
                    #if !os(tvOS)
                    .keyboardShortcut(.escape, modifiers: [])
                    #endif
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
                        .selectable(true)
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
                    #if !os(tvOS)
                    .keyboardShortcut(.escape, modifiers: [])
                    #endif
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
