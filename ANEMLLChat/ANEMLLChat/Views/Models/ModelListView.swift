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
        .frame(minWidth: 400, minHeight: 300)
    }

    // MARK: - Computed Properties

    private var hasErrorModels: Bool {
        modelManager.availableModels.contains { $0.downloadError != nil }
    }

    private var errorModels: [ModelInfo] {
        modelManager.availableModels.filter { $0.downloadError != nil }
    }

    // MARK: - Active Model Section

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
                    Text(model.name)
                        .font(.headline)

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

            // Show error message if any
            if let error = modelManager.errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
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
        NavigationStack {
            Form {
                Section {
                    TextField("HuggingFace Repo ID", text: $repoId)
                        .textContentType(.URL)
                        .autocorrectionDisabled()
                        #if os(iOS)
                        .textInputAutocapitalization(.never)
                        #endif

                    TextField("Display Name", text: $name)
                } header: {
                    Text("Custom Model")
                } footer: {
                    Text("Enter a HuggingFace repo ID like 'anemll/my-model'")
                }
            }
            .navigationTitle("Add Model")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button("Add") {
                        Task {
                            await modelManager.addCustomModel(repoId: repoId, name: name)
                            dismiss()
                        }
                    }
                    .disabled(repoId.isEmpty || name.isEmpty)
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
