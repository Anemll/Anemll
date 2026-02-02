//
//  ModelCard.swift
//  ANEMLLChat
//
//  Individual model card display
//

import SwiftUI
#if os(macOS)
import AppKit
#endif

struct ModelCard: View {
    let model: ModelInfo

    @Environment(ModelManagerViewModel.self) private var modelManager

    @State private var showingDeleteAlert = false
    @State private var showingModelDetail = false

    private var isLoaded: Bool {
        modelManager.loadedModelId == model.id
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Status icon - only show on macOS to save space on iPhone
            #if os(macOS)
            statusIcon
            #endif

            // Model info - fixed layout for consistency
            VStack(alignment: .leading, spacing: 2) {
                // Row 1: Info button + Name - FULL WIDTH (spans above buttons)
                HStack(spacing: 4) {
                    Button {
                        showingModelDetail = true
                    } label: {
                        Image(systemName: "info.circle")
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.blue)

                    Text(model.name)
                        .font(.headline)
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                }
                .frame(height: 20)

                // Row 2: Description + Action buttons (buttons moved down)
                HStack(spacing: 6) {
                    Text(model.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)

                    Spacer(minLength: 0)

                    // Action buttons on same row as description
                    actionButton
                }
                .frame(height: 28)

                // Row 3: Metadata - compact format
                HStack(spacing: 6) {
                    Text(model.size)
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    if let context = model.contextLength {
                        Text("\(context)ctx")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    if let arch = model.architecture {
                        Text(arch)
                            .font(.caption2)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(Color.secondary.opacity(0.2), in: Capsule())
                    }

                    Spacer(minLength: 0)
                }
                .frame(height: 18)
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 4)
        .contentShape(Rectangle())
        .onTapGesture {
            handleTap()
        }
        .contextMenu {
            contextMenuItems
        }
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
            if model.isDownloaded {
                Button(role: .destructive) {
                    showingDeleteAlert = true
                } label: {
                    Label("Delete", systemImage: "trash")
                }

                #if os(macOS)
                Button {
                    showInFinder()
                } label: {
                    Label("Finder", systemImage: "folder")
                }
                .tint(.blue)
                #endif
            }
        }
        .alert("Delete Model", isPresented: $showingDeleteAlert) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                Task {
                    await modelManager.deleteModel(model)
                }
            }
        } message: {
            Text("Are you sure you want to delete \(model.name)? This will remove all downloaded files.")
        }
        .sheet(isPresented: $showingModelDetail) {
            ModelDetailView(model: model)
        }
    }

    // MARK: - Status Icon

    @ViewBuilder
    private var statusIcon: some View {
        ZStack {
            Circle()
                .fill(statusBackground)
                .frame(width: 44, height: 44)

            Image(systemName: model.statusIcon)
                .font(.title3)
                .foregroundStyle(statusForeground)
        }
    }

    private var statusBackground: Color {
        switch model.status {
        case .available: return .blue.opacity(0.15)
        case .downloading: return .orange.opacity(0.15)
        case .downloaded: return .green.opacity(0.15)
        case .error: return .red.opacity(0.15)
        }
    }

    private var statusForeground: Color {
        switch model.status {
        case .available: return .blue
        case .downloading: return .orange
        case .downloaded: return .green
        case .error: return .red
        }
    }

    // MARK: - Action Button

    @ViewBuilder
    private var actionButton: some View {
        switch model.status {
        case .available:
            Button {
                Task {
                    await modelManager.downloadModel(model)
                }
            } label: {
                Image(systemName: "arrow.down.circle")
                    .font(.title2)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.blue)

        case .downloading:
            ProgressView()
                .controlSize(.small)

        case .downloaded:
            if modelManager.loadingModelId == model.id {
                // Animated loading indicator - more visible
                ModelLoadingIndicator()
            } else {
                HStack(spacing: 6) {
                    // Delete button
                    Button {
                        showingDeleteAlert = true
                    } label: {
                        Image(systemName: "trash")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                    .controlSize(.small)

                    // Load button
                    Button {
                        Task {
                            await modelManager.loadModelForInference(model)
                        }
                    } label: {
                        Text(isLoaded ? "Loaded" : "Load")
                            .font(.caption2)
                            .fontWeight(.medium)
                            .fixedSize()  // Prevent text wrapping
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(isLoaded ? .green : .blue)
                    .controlSize(.small)
                    .disabled(isLoaded)
                }
            }

        case .error(let message):
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
            .help(message)
        }
    }

    // MARK: - Context Menu

    @ViewBuilder
    private var contextMenuItems: some View {
        if model.isDownloaded {
            Button {
                Task {
                    await modelManager.loadModelForInference(model)
                }
            } label: {
                Label("Load Model", systemImage: "cpu")
            }
            .disabled(isLoaded)

            Divider()

            #if os(macOS)
            Button {
                showInFinder()
            } label: {
                Label("Show in Finder", systemImage: "folder")
            }

            Divider()
            #endif

            Button(role: .destructive) {
                showingDeleteAlert = true
            } label: {
                Label("Delete", systemImage: "trash")
            }
        } else {
            Button {
                Task {
                    await modelManager.downloadModel(model)
                }
            } label: {
                Label("Download", systemImage: "arrow.down.circle")
            }
        }
    }

    // MARK: - Show in Finder

    private func showInFinder() {
        #if os(macOS)
        guard let path = model.localPath else { return }
        let url = URL(fileURLWithPath: path)
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: url.path)
        #endif
    }

    // MARK: - Actions

    private func handleTap() {
        switch model.status {
        case .downloaded:
            if !isLoaded && !modelManager.isLoadingModel {
                Task {
                    await modelManager.loadModelForInference(model)
                }
            }
        case .available:
            Task {
                await modelManager.downloadModel(model)
            }
        default:
            break
        }
    }
}

#Preview {
    List {
        ModelCard(model: ModelInfo(
            id: "test/model-1",
            name: "Test Model",
            description: "A test model for preview",
            size: "1.2 GB",
            contextLength: 512,
            architecture: "llama",
            isDownloaded: true
        ))
        .environment(ModelManagerViewModel())

        ModelCard(model: ModelInfo(
            id: "test/model-2",
            name: "Another Model",
            description: "Available for download",
            size: "2.5 GB",
            contextLength: 1024,
            architecture: "qwen"
        ))
        .environment(ModelManagerViewModel())
    }
}

// MARK: - Model Detail View

import Yams

/// Parsed model configuration from meta.yaml
struct ModelMetadata: Sendable {
    let version: String
    let modelType: String
    let modelPrefix: String
    let contextLength: Int
    let batchSize: Int
    let lutFFN: Int?
    let lutLMHead: Int?
    let lutEmbeddings: Int?
    let numChunks: Int
    let splitLMHead: Int
    let argmaxInModel: Bool
    let slidingWindow: Int?

    static func load(from path: String) -> ModelMetadata? {
        guard FileManager.default.fileExists(atPath: path),
              let content = try? String(contentsOfFile: path, encoding: .utf8),
              let yaml = try? Yams.load(yaml: content) as? [String: Any],
              let modelInfo = yaml["model_info"] as? [String: Any],
              let params = modelInfo["parameters"] as? [String: Any] else {
            return nil
        }

        return ModelMetadata(
            version: modelInfo["version"] as? String ?? "Unknown",
            modelType: modelInfo["model_type"] as? String ?? "chunked",
            modelPrefix: params["model_prefix"] as? String ?? "unknown",
            contextLength: params["context_length"] as? Int ?? 2048,
            batchSize: params["batch_size"] as? Int ?? 32,
            lutFFN: params["lut_ffn"] as? Int,
            lutLMHead: params["lut_lmhead"] as? Int,
            lutEmbeddings: params["lut_embeddings"] as? Int,
            numChunks: params["num_chunks"] as? Int ?? 1,
            splitLMHead: params["split_lm_head"] as? Int ?? 8,
            argmaxInModel: params["argmax_in_model"] as? Bool ?? false,
            slidingWindow: params["sliding_window"] as? Int
        )
    }
}

struct ModelDetailView: View {
    let model: ModelInfo
    @Environment(\.dismiss) private var dismiss

    @State private var metadata: ModelMetadata?
    @State private var isLoading = true

    var body: some View {
        NavigationStack {
            List {
                // Basic Info Section
                Section("Model Information") {
                    DetailRow(label: "Name", value: model.name)
                    DetailRow(label: "ID", value: model.id)
                    DetailRow(label: "Size", value: model.size)
                    if let arch = model.architecture {
                        DetailRow(label: "Architecture", value: arch.capitalized)
                    }
                    DetailRow(label: "Status", value: statusText)
                }

                // Meta.yaml Details (if available)
                if let meta = metadata {
                    Section("Configuration") {
                        DetailRow(label: "Version", value: meta.version)
                        DetailRow(label: "Model Type", value: meta.modelType.capitalized)
                        DetailRow(label: "Model Prefix", value: meta.modelPrefix)
                    }

                    Section("Parameters") {
                        DetailRow(label: "Context Length", value: "\(meta.contextLength) tokens")
                        DetailRow(label: "Batch Size", value: "\(meta.batchSize)")
                        DetailRow(label: "Chunks", value: "\(meta.numChunks)")
                        DetailRow(label: "Split LM Head", value: "\(meta.splitLMHead)")

                        if let sw = meta.slidingWindow {
                            DetailRow(label: "Sliding Window", value: "\(sw)")
                        }

                        if meta.argmaxInModel {
                            DetailRow(label: "Argmax in Model", value: "Yes")
                        }
                    }

                    Section("Quantization (LUT bits)") {
                        if let lut = meta.lutFFN, lut > 0 {
                            DetailRow(label: "FFN", value: "\(lut)-bit")
                        } else {
                            DetailRow(label: "FFN", value: "None")
                        }

                        if let lut = meta.lutLMHead, lut > 0 {
                            DetailRow(label: "LM Head", value: "\(lut)-bit")
                        } else {
                            DetailRow(label: "LM Head", value: "None")
                        }

                        if let lut = meta.lutEmbeddings, lut > 0 {
                            DetailRow(label: "Embeddings", value: "\(lut)-bit")
                        } else {
                            DetailRow(label: "Embeddings", value: "None")
                        }
                    }
                } else if model.isDownloaded && !isLoading {
                    Section {
                        Text("Could not load model configuration")
                            .foregroundStyle(.secondary)
                    }
                } else if !model.isDownloaded {
                    Section {
                        Text("Download the model to view detailed configuration")
                            .foregroundStyle(.secondary)
                    }
                }

                // Local Path (if downloaded)
                if let path = model.localPath {
                    Section("Storage") {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Local Path")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(path)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Model Details")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .task {
                await loadMetadata()
            }
        }
    }

    private var statusText: String {
        switch model.status {
        case .available: return "Available for download"
        case .downloading: return "Downloading..."
        case .downloaded: return "Downloaded"
        case .error(let msg): return "Error: \(msg)"
        }
    }

    private func loadMetadata() async {
        isLoading = true
        defer { isLoading = false }

        guard model.isDownloaded,
              let localPath = model.localPath else {
            return
        }

        let metaPath = URL(fileURLWithPath: localPath)
            .appendingPathComponent("meta.yaml")
            .path

        metadata = ModelMetadata.load(from: metaPath)
    }
}

// MARK: - Detail Row

private struct DetailRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
        }
    }
}

// MARK: - Model Loading Indicator

/// Animated loading indicator for model loading state
private struct ModelLoadingIndicator: View {
    @State private var isAnimating = false
    @State private var rotation: Double = 0

    var body: some View {
        HStack(spacing: 6) {
            // Animated brain/chip icon
            ZStack {
                // Pulsing background
                Circle()
                    .fill(Color.green.opacity(0.2))
                    .frame(width: 28, height: 28)
                    .scaleEffect(isAnimating ? 1.2 : 0.8)
                    .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isAnimating)

                // Rotating gear
                Image(systemName: "cpu")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.green)
                    .rotationEffect(.degrees(rotation))
            }

            Text("Loading...")
                .font(.caption)
                .fontWeight(.medium)
                .foregroundStyle(.green)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.green.opacity(0.1), in: Capsule())
        .onAppear {
            isAnimating = true
            withAnimation(.linear(duration: 2).repeatForever(autoreverses: false)) {
                rotation = 360
            }
        }
    }
}
