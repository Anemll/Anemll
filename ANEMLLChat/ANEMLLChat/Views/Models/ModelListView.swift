//
//  ModelListView.swift
//  ANEMLLChat
//
//  Model browser and download manager
//

import SwiftUI
import UniformTypeIdentifiers
#if os(iOS)
import UIKit
#else
import AppKit
#endif

struct ModelListView: View {
    @Environment(ModelManagerViewModel.self) private var modelManager
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(\.dismiss) private var dismiss

    @State private var showingAddModel = false
    @State private var scrollProxy: ScrollViewProxy?
    @State private var isRefreshing = false
    @State private var showRefreshConfirmation = false
    @State private var refreshResultText = ""

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Button("Done") { dismiss() }
                Spacer()
                Text("Models").font(.headline)
                Spacer()
                Button { showingAddModel = true } label: {
                    Label("Add Model", systemImage: "plus")
                }
            }
            .padding()
            #if os(iOS)
            .padding(.horizontal, 4)
            #endif

            ScrollViewReader { proxy in
            List {
                // Active model (if any is loaded)
                if let loadedId = modelManager.loadedModelId,
                   let loadedModel = modelManager.availableModels.first(where: { $0.id == loadedId }) {
                    activeModelSection(loadedModel)
                }

                // Currently downloading (most important - user needs to see progress)
                if let downloadingId = modelManager.downloadingModelId,
                   let model = modelManager.availableModels.first(where: { $0.id == downloadingId }) {
                    downloadingSection(model)
                }

                // Downloaded models (ready to load)
                if !modelManager.downloadedModels.isEmpty {
                    downloadedSection
                }

                // Available for download
                if !modelManager.availableForDownload.isEmpty {
                    availableSection
                }

                // Models with errors
                if hasErrorModels {
                    errorSection
                }

                // Storage info
                storageSection
            }
            #if os(iOS)
            .listStyle(.insetGrouped)
            .contentMargins(.horizontal, 16, for: .scrollContent)
            #else
            .listStyle(.inset)
            #endif
            .refreshable {
                await modelManager.refreshModelStatus()
            }
            .task {
                // Log model state when view appears
                print("[ModelListView] task: \(modelManager.availableModels.count) models")
                logInfo("ModelListView task: \(modelManager.availableModels.count) total", category: .model)

                if modelManager.availableModels.isEmpty {
                    print("[ModelListView] empty, calling loadModels")
                    await modelManager.loadModels()
                }
            }
            .onAppear {
                scrollProxy = proxy
            }
            // Scroll to downloading section when download starts
            .onChange(of: modelManager.downloadingModelId) { oldValue, newValue in
                if newValue != nil && oldValue == nil {
                    // Download just started - scroll to show it
                    withAnimation(.easeOut(duration: 0.3)) {
                        proxy.scrollTo("downloading", anchor: .top)
                    }
                }
            }
            .onChange(of: modelManager.justAddedModelId) { _, newValue in
                guard let modelId = newValue else { return }
                withAnimation(.easeOut(duration: 0.3)) {
                    proxy.scrollTo(modelId, anchor: .center)
                }
            }
            } // End ScrollViewReader
        }
        .sheet(isPresented: $showingAddModel) {
            AddModelView()
                .environment(modelManager)
        }
        #if os(macOS)
        .frame(minWidth: 720, maxWidth: .infinity, minHeight: 560, maxHeight: .infinity)
        #endif
        // Auto-dismiss when a model is loaded
        .onChange(of: modelManager.loadedModelId) { oldValue, newValue in
            if newValue != nil && oldValue != newValue {
                dismiss()
            }
        }
        // Stay in Models view during download - user can manually dismiss when ready
        // Error toast
        .errorToast(Binding(
            get: { modelManager.errorMessage },
            set: { modelManager.errorMessage = $0 }
        ))
        // Weight size warning alert
        // Note: Don't call cancelLoadModel in the binding setter - it clears pendingLoadModel
        // before the button action runs. Only cancel explicitly via the Cancel button.
        .alert("Large Weight Files", isPresented: Binding(
            get: { modelManager.showWeightWarningAlert },
            set: { modelManager.showWeightWarningAlert = $0 }
        )) {
            Button("Cancel", role: .cancel) {
                modelManager.cancelLoadModel()
            }
            Button("Load Anyway") {
                Task {
                    await modelManager.confirmLoadModel()
                }
            }
        } message: {
            Text(modelManager.weightWarningMessage ?? "This model has weight files that may not load correctly on this device.")
        }
        // [ANE-COMPAT:M1-A14] Compatibility warning alert for Gemma global attention (load-time)
        .alert("Device Compatibility", isPresented: Binding(
            get: { modelManager.showCompatibilityWarningAlert },
            set: { modelManager.showCompatibilityWarningAlert = $0 }
        )) {
            Button("Cancel", role: .cancel) {
                modelManager.cancelLoadModel()
            }
            Button("Load Anyway") {
                Task {
                    await modelManager.confirmLoadModel()
                }
            }
        } message: {
            Text(modelManager.compatibilityWarningMessage ?? "This model may not work correctly on this device.")
        }
        // [ANE-COMPAT:M1-A14] Pre-download compatibility warning alert for Gemma global attention
        .alert("Device Compatibility", isPresented: Binding(
            get: { modelManager.showDownloadCompatibilityWarningAlert },
            set: { modelManager.showDownloadCompatibilityWarningAlert = $0 }
        )) {
            Button("Cancel", role: .cancel) {
                modelManager.cancelDownloadModel()
            }
            Button("Download Anyway") {
                Task {
                    await modelManager.confirmDownloadModel()
                }
            }
        } message: {
            Text(modelManager.downloadCompatibilityWarningMessage ?? "This model may not work correctly on this device.")
        }
    }

    // MARK: - Computed Properties

    private var hasErrorModels: Bool {
        modelManager.availableModels.contains { $0.downloadError != nil }
    }

    private var errorModels: [ModelInfo] {
        modelManager.availableModels.filter { $0.downloadError != nil }
    }

    // MARK: - Active Model Section

    @State private var showingActiveModelDetail = false

    private func activeModelSection(_ model: ModelInfo) -> some View {
        Section {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(Color.green.opacity(0.15))
                        .frame(width: 44, height: 44)

                    Image(systemName: "bolt.fill")
                        .font(.title3)
                        .foregroundStyle(.green)
                }

                VStack(alignment: .leading, spacing: 4) {
                    // Name with info button
                    Button {
                        showingActiveModelDetail = true
                    } label: {
                        HStack(spacing: 4) {
                            Text(model.name)
                                .font(.headline)
                                .foregroundStyle(.primary)

                            Image(systemName: "info.circle")
                                .font(.caption)
                                .foregroundStyle(.blue)
                        }
                    }
                    .buttonStyle(.plain)

                    Text("Loaded & Active")
                        .font(.caption)
                        .foregroundStyle(.green)
                }

                Spacer()

                Button {
                    Task {
                        await modelManager.unloadCurrentModel()
                    }
                } label: {
                    Text("Unload")
                        .font(.caption)
                        .fontWeight(.medium)
                }
                .buttonStyle(.bordered)
                .tint(.orange)
            }
            .padding(.vertical, 4)
            .sheet(isPresented: $showingActiveModelDetail) {
                ModelDetailView(model: model)
            }
        } header: {
            Label("Active Model", systemImage: "bolt.circle.fill")
                .foregroundStyle(.green)
        }
    }

    // MARK: - Downloaded Section

    private var downloadedSection: some View {
        Section {
            ForEach(modelManager.downloadedModels.filter { $0.id != modelManager.loadedModelId }) { model in
                ModelCard(model: model)
                    .environment(modelManager)
                    .id(model.id)
            }
        } header: {
            Text("Downloaded")
        } footer: {
            Text("Tap a model to load it for chat.")
        }
    }

    // MARK: - Available Section

    private var availableSection: some View {
        Section {
            ForEach(modelManager.availableForDownload) { model in
                ModelCard(model: model)
                    .environment(modelManager)
                    .id(model.id)
            }
        } header: {
            HStack {
                Text("Available")
                Spacer()
                Button {
                    Task {
                        isRefreshing = true
                        await modelManager.refreshModelStatus()
                        // Brief delay so the spinner is visible
                        try? await Task.sleep(for: .milliseconds(300))
                        isRefreshing = false
                        refreshResultText = modelManager.lastRefreshDiscoveredCount > 0
                            ? "+\(modelManager.lastRefreshDiscoveredCount) new"
                            : "Up to date"
                        showRefreshConfirmation = true
                        // Auto-hide confirmation
                        try? await Task.sleep(for: .seconds(2.5))
                        showRefreshConfirmation = false
                    }
                } label: {
                    HStack(spacing: 4) {
                        if isRefreshing {
                            ProgressView()
                                .controlSize(.mini)
                            Text("Checking...")
                                .font(.caption)
                        } else if showRefreshConfirmation {
                            Image(systemName: modelManager.lastRefreshDiscoveredCount > 0
                                  ? "plus.circle.fill" : "checkmark.circle.fill")
                                .font(.caption)
                                .foregroundStyle(modelManager.lastRefreshDiscoveredCount > 0 ? .blue : .green)
                            Text(refreshResultText)
                                .font(.caption)
                                .foregroundStyle(modelManager.lastRefreshDiscoveredCount > 0 ? .blue : .green)
                        } else {
                            Image(systemName: "arrow.clockwise")
                                .font(.caption)
                            Text("Refresh")
                                .font(.caption)
                        }
                    }
                }
                .buttonStyle(.plain)
                .foregroundStyle(.orange)
                .disabled(isRefreshing)
            }
        } footer: {
            Text("Download models from HuggingFace. Tap \(Image(systemName: "arrow.clockwise")) to refresh list.")
        }
    }

    // MARK: - Downloading Section

    private func downloadingSection(_ model: ModelInfo) -> some View {
        Section {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text(model.name)
                        .font(.headline)

                    Spacer()

                    Button {
                        Task {
                            await modelManager.cancelDownload()
                        }
                    } label: {
                        Text("Cancel")
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.plain)
                    .contentShape(Rectangle())
                }

                if let progress = modelManager.downloadProgress {
                    DownloadProgressView(progress: progress)
                } else {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text("Preparing download...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.vertical, 4)
            .contentShape(Rectangle()) // Prevent taps on empty space from propagating
            .allowsHitTesting(true)
        } header: {
            HStack {
                Text("Downloading")
                Spacer()
            }
        }
        .id("downloading") // For scroll targeting
    }

    // MARK: - Error Section

    private var errorSection: some View {
        Section {
            ForEach(errorModels) { model in
                HStack(spacing: 12) {
                    ZStack {
                        Circle()
                            .fill(Color.red.opacity(0.15))
                            .frame(width: 44, height: 44)

                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.title3)
                            .foregroundStyle(.red)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Text(model.name)
                            .font(.headline)

                        if let error = model.downloadError {
                            Text(error)
                                .font(.caption)
                                .foregroundStyle(.red)
                                .lineLimit(2)
                        }

                        // Show the broken path for linked/imported models so user can identify it
                        if model.sourceKind != .huggingFace,
                           let path = model.linkedPath ?? model.localPath {
                            HStack(spacing: 4) {
                                Text(path)
                                    .font(.system(.caption2, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)

                                Button {
                                    #if os(macOS)
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.setString(path, forType: .string)
                                    #else
                                    UIPasteboard.general.string = path
                                    #endif
                                } label: {
                                    Image(systemName: "doc.on.doc")
                                        .font(.caption2)
                                }
                                .buttonStyle(.plain)
                                .foregroundStyle(.secondary)
                                .help("Copy path")
                            }
                        }
                    }

                    Spacer()

                    if model.sourceKind == .huggingFace {
                        Button {
                            Task {
                                await modelManager.downloadModel(model)
                            }
                        } label: {
                            Image(systemName: "arrow.clockwise")
                                .font(.title2)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.orange)
                    } else {
                        // Remove button for broken linked/imported models
                        Button(role: .destructive) {
                            Task {
                                await modelManager.deleteModel(model)
                            }
                        } label: {
                            Image(systemName: "trash")
                                .font(.title2)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.red)
                    }
                }
                .padding(.vertical, 4)
                .id(model.id)
            }
        } header: {
            Label("Failed Downloads", systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
        } footer: {
            Text("Tap retry for failed downloads. Linked models require re-linking if the source folder moved.")
        }
    }

    // MARK: - Storage Section

    private var storageSection: some View {
        Section {
            HStack {
                Label("Downloaded Models", systemImage: "internaldrive")
                Spacer()
                Text(modelManager.downloadedModelsSize)
                    .foregroundStyle(.secondary)
            }

            // Device info
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(DeviceType.chipName)
                        .font(.caption)
                        .fontWeight(.medium)
                    Spacer()
                    Text(DeviceType.physicalMemoryString)
                        .font(.caption)
                }
                HStack {
                    Text(DeviceType.osVersionString)
                        .font(.caption)
                    Spacer()
                    Text("\(DeviceType.processorCount) cores")
                        .font(.caption)
                }
            }
            .foregroundStyle(.secondary)

            // Debug info - always show model counts
            HStack {
                Text("Total: \(modelManager.availableModels.count)")
                    .font(.caption)
                Spacer()
                Text("Available: \(modelManager.availableForDownload.count)")
                    .font(.caption)
                Spacer()
                Text("Downloaded: \(modelManager.downloadedModels.count)")
                    .font(.caption)
            }
            .foregroundStyle(.secondary)

            // Error messages now shown as toast at top of view
        } header: {
            Text("Storage")
        }
    }
}

// MARK: - Add Model View

struct AddModelView: View {
    @Environment(ModelManagerViewModel.self) private var modelManager
    @Environment(\.dismiss) private var dismiss

    @State private var repoId = ""
    @State private var hfDisplayName = ""
    @State private var isAdding = false
    @State private var addError: String?
    @State private var showSuccess = false

    #if os(macOS)
    @State private var sourceType: AddSourceType = .huggingFace
    @State private var selectedLocalFolder: URL?
    @State private var localInspection: LocalModelInspection?
    @State private var localDisplayName = ""
    @AppStorage("localModelImportModeSelection") private var localImportModeRawValue = LocalModelImportMode.importCopy.rawValue
    @State private var showingFolderPicker = false
    @State private var isDropTargeted = false
    #endif

    private enum AddSourceType: String, CaseIterable {
        case huggingFace
        #if os(macOS)
        case localFolder
        #endif

        var title: String {
            switch self {
            case .huggingFace:
                return "HuggingFace"
            #if os(macOS)
            case .localFolder:
                return "Local Folder"
            #endif
            }
        }
    }

    // Auto-generate display name from repo ID
    private var hfSuggestedName: String {
        guard !repoId.isEmpty else { return "" }

        // Extract model name from repo ID (e.g., "anemll/google-gemma-3-4b" -> "Gemma 3 4B")
        let parts = repoId.split(separator: "/")
        let modelPart = parts.count > 1 ? String(parts[1]) : repoId

        // Clean up the name
        var name = modelPart
            .replacingOccurrences(of: "anemll-", with: "")
            .replacingOccurrences(of: "google-", with: "")
            .replacingOccurrences(of: "-it-", with: "-")
            .replacingOccurrences(of: "-qat-", with: "-QAT-")
            .replacingOccurrences(of: "-int4", with: "")
            .replacingOccurrences(of: "-unquantized", with: "")
            .replacingOccurrences(of: "_0.3.5", with: "")
            .replacingOccurrences(of: "-ctx", with: " CTX")
            .replacingOccurrences(of: "-", with: " ")

        // Capitalize first letter of each word, handle special cases
        name = name.split(separator: " ").map { word in
            let w = String(word)
            if w.uppercased() == w { return w } // Keep all-caps (like QAT, CTX)
            if w.lowercased() == "gemma" { return "Gemma" }
            if w.lowercased() == "qwen" { return "Qwen" }
            if w.lowercased() == "llama" { return "LLaMA" }
            if w.lowercased() == "deepseek" { return "DeepSeek" }
            // Numbers and sizes
            if w.contains(where: { $0.isNumber }) { return w.uppercased() }
            return w.capitalized
        }.joined(separator: " ")

        return name
    }

    // Use suggested name if display name is empty
    private var hfEffectiveName: String {
        hfDisplayName.isEmpty ? hfSuggestedName : hfDisplayName
    }

    #if os(macOS)
    private var localImportMode: LocalModelImportMode {
        LocalModelImportMode(rawValue: localImportModeRawValue) ?? .importCopy
    }

    private var localImportModeBinding: Binding<LocalModelImportMode> {
        Binding(
            get: { localImportMode },
            set: { localImportModeRawValue = $0.rawValue }
        )
    }

    private var localEffectiveName: String {
        localDisplayName.isEmpty ? (localInspection?.suggestedDisplayName ?? "") : localDisplayName
    }
    #endif

    private var addButtonTitle: String {
        #if os(macOS)
        if sourceType == .localFolder {
            return localImportMode == .importCopy ? "Import Model" : "Link Model"
        }
        #endif
        return "Add Model"
    }

    private var addButtonDisabled: Bool {
        if isAdding { return true }
        #if os(macOS)
        if sourceType == .localFolder {
            return localInspection == nil || localEffectiveName.isEmpty
        }
        #endif
        return !isValidRepoId || hfEffectiveName.isEmpty
    }

    private var isValidRepoId: Bool {
        // Must contain a slash and have content on both sides
        let parts = repoId.split(separator: "/")
        return parts.count >= 2 && parts[0].count > 0 && parts[1].count > 0
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header with icon
            VStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(Color.blue.opacity(0.1))
                        .frame(width: 60, height: 60)

                    Image(systemName: "plus.circle.fill")
                        .font(.system(size: 30))
                        .foregroundStyle(.blue)
                }

                #if os(macOS)
                Text(sourceType == .huggingFace ? "Add Custom Model" : "Add Local Model")
                    .font(.title2)
                    .fontWeight(.semibold)

                Text(sourceType == .huggingFace ? "Add a model from HuggingFace to your library" : "Drop a compiled model folder, then import or link it")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                #else
                Text("Add Custom Model")
                    .font(.title2)
                    .fontWeight(.semibold)

                Text("Add a model from HuggingFace to your library")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                #endif

                #if os(macOS)
                Picker("Source", selection: $sourceType) {
                    ForEach(AddSourceType.allCases, id: \.self) { source in
                        Text(source.title).tag(source)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: sourceType) { _, _ in
                    addError = nil
                    showSuccess = false
                }
                #endif

                #if os(iOS)
                // Quick access to ANEMLL models
                Button {
                    openAnemllHuggingFace()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "link")
                            .font(.caption)
                        Text("Browse ANEMLL Models")
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                    .foregroundStyle(.blue)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.blue.opacity(0.1), in: Capsule())
                }
                .buttonStyle(.plain)
                #else
                if sourceType == .huggingFace {
                    Button {
                        openAnemllHuggingFace()
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "link")
                                .font(.caption)
                            Text("Browse ANEMLL Models")
                                .font(.caption)
                                .fontWeight(.medium)
                        }
                        .foregroundStyle(.blue)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.blue.opacity(0.1), in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
                #endif
            }
            .padding(.top, 24)
            .padding(.bottom, 20)

            // Form
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    #if os(macOS)
                    if sourceType == .localFolder {
                        localFolderForm
                    } else {
                        huggingFaceForm
                    }
                    #else
                    huggingFaceForm
                    #endif

                    // Error message
                    if let error = addError {
                        HStack(spacing: 6) {
                            Image(systemName: "exclamationmark.triangle.fill")
                            Text(error)
                        }
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                    }

                    // Success message
                    if showSuccess {
                        HStack(spacing: 6) {
                            Image(systemName: "checkmark.circle.fill")
                            #if os(macOS)
                            Text(sourceType == .localFolder ? "Model added successfully." : "Model added! Download starting...")
                            #else
                            Text("Model added! Download starting...")
                            #endif
                        }
                        .font(.caption)
                        .foregroundStyle(.green)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.green.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 4)
            }

            // Buttons
            HStack(spacing: 12) {
                Button {
                    dismiss()
                } label: {
                    Text("Cancel")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .keyboardShortcut(.cancelAction)

                Button {
                    addModel()
                } label: {
                    HStack(spacing: 6) {
                        if isAdding {
                            ProgressView()
                                .controlSize(.small)
                        }
                        Text(isAdding ? "Adding..." : addButtonTitle)
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(addButtonDisabled)
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 24)
            .padding(.top, 16)
        }
        #if os(macOS)
        .fileImporter(
            isPresented: $showingFolderPicker,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                inspectLocalFolder(url)
            case .failure(let error):
                addError = error.localizedDescription
            }
        }
        #endif
        #if os(macOS)
        .frame(width: 520, height: 700)
        #else
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        #endif
    }

    private var huggingFaceForm: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                Text("HuggingFace Repository")
                    .font(.subheadline)
                    .fontWeight(.medium)

                TextField("anemll/model-name", text: $repoId)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    #if os(iOS)
                    .textInputAutocapitalization(.never)
                    #endif
                    .onChange(of: repoId) { _, _ in
                        addError = nil
                    }

                if !repoId.isEmpty && !isValidRepoId {
                    Label("Format: owner/model-name", systemImage: "exclamationmark.circle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                } else {
                    Text("Example: anemll/anemll-google-gemma-3-4b-it-qat-int4-unquantized-ctx4096_0.3.5")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Display Name")
                    .font(.subheadline)
                    .fontWeight(.medium)

                TextField(hfSuggestedName.isEmpty ? "Model Name" : hfSuggestedName, text: $hfDisplayName)
                    .textFieldStyle(.roundedBorder)

                if !hfSuggestedName.isEmpty && hfDisplayName.isEmpty {
                    HStack(spacing: 4) {
                        Image(systemName: "sparkles")
                            .font(.caption)
                        Text("Auto-suggested: \(hfSuggestedName)")
                            .font(.caption)
                    }
                    .foregroundStyle(.blue)
                } else {
                    Text("Friendly name shown in the model list")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    #if os(macOS)
    private var localFolderForm: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Model Folder")
                .font(.subheadline)
                .fontWeight(.medium)

            VStack(spacing: 10) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(isDropTargeted ? Color.blue : Color.secondary.opacity(0.3), style: StrokeStyle(lineWidth: 1.5, dash: [6]))
                        .background((isDropTargeted ? Color.blue.opacity(0.08) : Color.secondary.opacity(0.06)), in: RoundedRectangle(cornerRadius: 10))

                    VStack(spacing: 6) {
                        Image(systemName: "tray.and.arrow.down")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                        Text("Drop model folder here")
                            .font(.subheadline)
                        Text("Expected: contains meta.yaml and .mlmodelc")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(12)
                }
                .frame(height: 120)
                .onDrop(of: [UTType.fileURL], isTargeted: $isDropTargeted) { providers in
                    handleFolderDrop(providers)
                }

                Button {
                    showingFolderPicker = true
                } label: {
                    Label("Choose Folder...", systemImage: "folder")
                }
                .buttonStyle(.bordered)
            }

            if let selectedLocalFolder {
                Text("Dropped: \(selectedLocalFolder.path)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            if let localInspection {
                Text("Detected model root: \(localInspection.modelRootURL.path)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)

                VStack(alignment: .leading, spacing: 8) {
                    Text("Display Name")
                        .font(.subheadline)
                        .fontWeight(.medium)

                    TextField(localInspection.suggestedDisplayName, text: $localDisplayName)
                        .textFieldStyle(.roundedBorder)

                    if localDisplayName.isEmpty {
                        Text("Auto-suggested: \(localInspection.suggestedDisplayName)")
                            .font(.caption)
                            .foregroundStyle(.blue)
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("Ingest Mode")
                        .font(.subheadline)
                        .fontWeight(.medium)

                    Picker("Ingest Mode", selection: localImportModeBinding) {
                        ForEach(LocalModelImportMode.allCases, id: \.self) { mode in
                            Text(mode.displayTitle).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)

                    Text(localImportMode == .importCopy ? "Copies files into app storage (portable)." : "Keeps external folder reference (faster setup).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
    #endif

    private func openAnemllHuggingFace() {
        if let url = URL(string: "https://huggingface.co/anemll") {
            #if os(iOS)
            UIApplication.shared.open(url)
            #else
            NSWorkspace.shared.open(url)
            #endif
        }
    }

    private func addModel() {
        isAdding = true
        addError = nil

        Task {
            #if os(macOS)
            if sourceType == .localFolder {
                guard let folder = selectedLocalFolder else {
                    await MainActor.run {
                        isAdding = false
                        addError = "Please drop or select a model folder first."
                    }
                    return
                }
                await modelManager.addLocalModel(from: folder, displayName: localEffectiveName, mode: localImportMode)
            } else {
                guard isValidRepoId else {
                    await MainActor.run {
                        isAdding = false
                        addError = "Please enter a valid HuggingFace repo ID"
                    }
                    return
                }
                await modelManager.addCustomModel(repoId: repoId.trimmingCharacters(in: .whitespaces), name: hfEffectiveName)
            }
            #else
            guard isValidRepoId else {
                await MainActor.run {
                    isAdding = false
                    addError = "Please enter a valid HuggingFace repo ID"
                }
                return
            }
            await modelManager.addCustomModel(repoId: repoId.trimmingCharacters(in: .whitespaces), name: hfEffectiveName)
            #endif

            await MainActor.run {
                isAdding = false

                // Check if there was an error (model already exists, etc.)
                if let error = modelManager.errorMessage, !error.isEmpty {
                    addError = error
                    modelManager.errorMessage = nil
                } else {
                    // Success - show feedback briefly then dismiss
                    showSuccess = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                        dismiss()
                    }
                }
            }
        }
    }

    #if os(macOS)
    private func inspectLocalFolder(_ url: URL) {
        // fileImporter returns a security-scoped URL — start access before any file I/O.
        let accessGranted = url.startAccessingSecurityScopedResource()
        defer { if accessGranted { url.stopAccessingSecurityScopedResource() } }
        selectedLocalFolder = url
        do {
            let inspection = try modelManager.inspectLocalModelFolder(url)
            localInspection = inspection
            localDisplayName = inspection.suggestedDisplayName
            addError = nil
        } catch {
            localInspection = nil
            addError = error.localizedDescription
        }
    }

    private func handleFolderDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }) else {
            return false
        }

        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
            if let data = item as? Data, let url = URL(dataRepresentation: data, relativeTo: nil) {
                Task { @MainActor in
                    inspectLocalFolder(url)
                }
                return
            }
            if let url = item as? URL {
                Task { @MainActor in
                    inspectLocalFolder(url)
                }
                return
            }
            if let nsURL = item as? NSURL, let url = nsURL as URL? {
                Task { @MainActor in
                    inspectLocalFolder(url)
                }
            }
        }
        return true
    }
    #endif
}

#Preview {
    ModelListView()
        .environment(ModelManagerViewModel())
        .environment(ChatViewModel())
}
