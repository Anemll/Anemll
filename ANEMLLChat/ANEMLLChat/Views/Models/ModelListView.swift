//
//  ModelListView.swift
//  ANEMLLChat
//
//  Model browser and download manager
//

import SwiftUI

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
    @State private var name = ""

    var body: some View {
        VStack(spacing: 16) {
            // Header
            Text("Add Model")
                .font(.headline)
                .padding(.top)

            // Form content
            VStack(alignment: .leading, spacing: 12) {
                Text("Custom Model")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                TextField("HuggingFace Repo ID", text: $repoId)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()

                TextField("Display Name", text: $name)
                    .textFieldStyle(.roundedBorder)

                Text("Enter a HuggingFace repo ID like 'anemll/my-model'")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal)

            Spacer()

            // Buttons
            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Add") {
                    Task {
                        await modelManager.addCustomModel(repoId: repoId, name: name)
                        dismiss()
                    }
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .disabled(repoId.isEmpty || name.isEmpty)
            }
            .padding()
        }
        .frame(width: 350, height: 220)
    }
}

#Preview {
    ModelListView()
        .environment(ModelManagerViewModel())
        .environment(ChatViewModel())
}
