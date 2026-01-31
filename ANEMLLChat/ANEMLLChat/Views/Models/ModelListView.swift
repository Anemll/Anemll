//
//  ModelListView.swift
//  ANEMLLChat
//
//  Model browser and download manager
//

import SwiftUI
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
            .contentMargins(.horizontal, 8, for: .scrollContent)
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
        }
        .sheet(isPresented: $showingAddModel) {
            AddModelView()
                .environment(modelManager)
        }
        #if os(macOS)
        .frame(minWidth: 400, minHeight: 300)
        #endif
        // Auto-dismiss when a model is loaded
        .onChange(of: modelManager.loadedModelId) { oldValue, newValue in
            if newValue != nil && oldValue != newValue {
                dismiss()
            }
        }
        // Error toast
        .errorToast(Binding(
            get: { modelManager.errorMessage },
            set: { modelManager.errorMessage = $0 }
        ))
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
            }
        } header: {
            Text("Available")
        } footer: {
            Text("Download models from HuggingFace.")
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

                    Button("Cancel") {
                        Task {
                            await modelManager.cancelDownload()
                        }
                    }
                    .foregroundStyle(.red)
                }

                if let progress = modelManager.downloadProgress {
                    DownloadProgressView(progress: progress)
                }
            }
            .padding(.vertical, 4)
        } header: {
            Text("Downloading")
        }
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
                    }

                    Spacer()

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
                }
                .padding(.vertical, 4)
            }
        } header: {
            Label("Failed Downloads", systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
        } footer: {
            Text("Tap retry to download again.")
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
    @State private var displayName = ""
    @State private var isAdding = false
    @State private var addError: String?
    @State private var showSuccess = false

    // Auto-generate display name from repo ID
    private var suggestedName: String {
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
    private var effectiveName: String {
        displayName.isEmpty ? suggestedName : displayName
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

                Text("Add Custom Model")
                    .font(.title2)
                    .fontWeight(.semibold)

                Text("Add a model from HuggingFace to your library")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

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
            }
            .padding(.top, 24)
            .padding(.bottom, 20)

            // Form
            VStack(alignment: .leading, spacing: 20) {
                // Repo ID Field
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

                    // Validation hint
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

                // Display Name Field
                VStack(alignment: .leading, spacing: 8) {
                    Text("Display Name")
                        .font(.subheadline)
                        .fontWeight(.medium)

                    TextField(suggestedName.isEmpty ? "Model Name" : suggestedName, text: $displayName)
                        .textFieldStyle(.roundedBorder)

                    if !suggestedName.isEmpty && displayName.isEmpty {
                        HStack(spacing: 4) {
                            Image(systemName: "sparkles")
                                .font(.caption)
                            Text("Auto-suggested: \(suggestedName)")
                                .font(.caption)
                        }
                        .foregroundStyle(.blue)
                    } else {
                        Text("Friendly name shown in the model list")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

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
                        Text("Model added! Download starting...")
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

            Spacer()

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
                        Text(isAdding ? "Adding..." : "Add Model")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!isValidRepoId || effectiveName.isEmpty || isAdding)
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 24)
            .padding(.top, 16)
        }
        #if os(macOS)
        .frame(width: 420, height: 480)
        #else
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        #endif
    }

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
        guard isValidRepoId else {
            addError = "Please enter a valid HuggingFace repo ID"
            return
        }

        isAdding = true
        addError = nil

        Task {
            await modelManager.addCustomModel(repoId: repoId.trimmingCharacters(in: .whitespaces), name: effectiveName)

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
}

#Preview {
    ModelListView()
        .environment(ModelManagerViewModel())
        .environment(ChatViewModel())
}
