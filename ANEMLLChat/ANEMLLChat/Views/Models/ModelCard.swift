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
    @State private var showingCancelLoadAlert = false
    @State private var showingModelDetail = false
    @State private var recommendedSampling: RecommendedSampling?
    @State private var computedLocalSize: String?

    private var isLoaded: Bool {
        modelManager.loadedModelId == model.id
    }

    private var isJustAdded: Bool {
        modelManager.justAddedModelId == model.id
    }

    /// Check if model has oversized weights that WILL cause issues on THIS device
    private var hasWeightWarning: Bool {
        guard model.isDownloaded else { return false }
        return modelManager.hasOversizedWeights(for: model) && DeviceType.requiresWeightSizeLimit
    }

    /// Check if model has oversized weights (informational - for Mac users)
    private var hasLargeWeights: Bool {
        guard model.isDownloaded else { return false }
        return modelManager.hasOversizedWeights(for: model)
    }

    // [ANE-COMPAT:M1-A14] Compatibility tag for Gemma global attention
    // Shows for both downloaded models and pre-download-checked models (cached from meta.yaml fetch)
    private var hasCompatibilityWarning: Bool {
        return modelManager.getGlobalAttentionWarning(for: model) != nil
    }

    private var displaySize: String {
        computedLocalSize ?? model.size
    }

    private var deleteMessage: String {
        switch model.sourceKind {
        case .localLinked:
            return "Are you sure you want to remove \(model.name)? This only removes the app reference. The original linked folder is not deleted."
        case .localImported, .huggingFace:
            return "Are you sure you want to delete \(model.name)? This will remove local model files from app storage."
        }
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Status icon - only show on macOS to save space on iPhone
            #if os(macOS)
            statusIcon
            #endif

            // Model info - fixed layout for consistency
            VStack(alignment: .leading, spacing: 2) {
                // Row 1: Info button + Name + Warning - FULL WIDTH (spans above buttons)
                HStack(spacing: 4) {
                    Button {
                        showingModelDetail = true
                    } label: {
                        Image(systemName: "info.circle")
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.orange)

                    Text(model.name)
                        .font(.headline)
                        .foregroundStyle(.primary)
                        .lineLimit(1)

                    // Weight warning indicator - red for this device, orange for info on Mac
                    if hasWeightWarning {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.caption2)
                            .foregroundStyle(.red)
                            .help("Model has weight files >1GB which may not load on this device")
                    } else if hasLargeWeights {
                        // Informational indicator on Mac - model won't work on iPhone/iPad
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                            .help("Model has weight files >1GB - won't load on iPhone or non-M-series iPad")
                    }

                    // [ANE-COMPAT:M1-A14] "Not Compatible" tag for Gemma global attention on M1/A14
                    if hasCompatibilityWarning {
                        Label("Not Compatible", systemImage: "xmark.circle.fill")
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .foregroundStyle(.red)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.red.opacity(0.15), in: Capsule())
                            .help("Gemma global attention not supported on \(DeviceType.chipName) — requires M2+/A15+")
                    }

                    if isJustAdded {
                        Label("Just Added", systemImage: "sparkles")
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .foregroundStyle(.blue)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.blue.opacity(0.15), in: Capsule())
                    }
                }
                .frame(minHeight: 20, alignment: .leading)

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
                    Text(displaySize)
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

                    // Recommended sampling indicator
                    if let sampling = recommendedSampling {
                        HStack(spacing: 2) {
                            Image(systemName: "dice.fill")
                                .font(.system(size: 8))
                            Text(String(format: "%.1f", sampling.temperature))
                        }
                        .font(.caption2)
                        .foregroundStyle(.green)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Color.green.opacity(0.15), in: Capsule())
                        .help("Recommended: temp=\(String(format: "%.2f", sampling.temperature)), top_p=\(String(format: "%.2f", sampling.topP)), top_k=\(sampling.topK)")
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
                    Task {
                        await modelManager.shareModelToIOS(model)
                    }
                } label: {
                    Label("Share to iOS", systemImage: "square.and.arrow.up")
                }
                .tint(.blue)

                Button {
                    showInFinder()
                } label: {
                    Label("Finder", systemImage: "folder")
                }
                .tint(.orange)
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
            Text(deleteMessage)
        }
        .sheet(isPresented: $showingModelDetail) {
            ModelDetailView(model: model)
        }
        .alert("Stop Loading?", isPresented: $showingCancelLoadAlert) {
            Button("Cancel", role: .cancel) {}
            Button("Stop", role: .destructive) {
                Task {
                    await modelManager.cancelModelLoading()
                }
            }
        } message: {
            Text("The model will be unloaded and loading will stop.")
        }
        .task {
            // Load recommended sampling for downloaded models
            if model.isDownloaded {
                modelManager.withLinkedModelAccess(for: model) { modelURL in
                    let metaPath = modelURL.appendingPathComponent("meta.yaml").path
                    if let metadata = ModelMetadata.load(from: metaPath) {
                        recommendedSampling = metadata.recommendedSampling
                    }
                }
            }
            // Compute actual size for local models that show "Local"
            if model.size == "Local" {
                computedLocalSize = modelManager.formattedModelSize(for: model)
            }
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

            // Warning badge overlay for models with large weights or compatibility issues
            if hasCompatibilityWarning {
                // [ANE-COMPAT:M1-A14] Badge for incompatible models
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 14))
                    .foregroundStyle(.red)
                    .background(
                        Circle()
                            .fill(backgroundCircleColor)
                            .frame(width: 18, height: 18)
                    )
                    .offset(x: 14, y: 14)
            } else if hasWeightWarning || hasLargeWeights {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 14))
                    .foregroundStyle(hasWeightWarning ? .red : .orange)
                    .background(
                        Circle()
                            .fill(backgroundCircleColor)
                            .frame(width: 18, height: 18)
                    )
                    .offset(x: 14, y: 14)
            }
        }
    }

    private var statusBackground: Color {
        switch model.status {
        case .available: return .orange.opacity(0.15)
        case .downloading: return .yellow.opacity(0.15)
        case .downloaded: return .green.opacity(0.15)
        case .error: return .red.opacity(0.15)
        }
    }

    private var statusForeground: Color {
        switch model.status {
        case .available: return .orange
        case .downloading: return .yellow
        case .downloaded: return .green
        case .error: return .red
        }
    }

    private var backgroundCircleColor: Color {
        #if os(macOS)
        return Color(NSColor.windowBackgroundColor)
        #else
        return Color(uiColor: .systemBackground)
        #endif
    }

    // MARK: - Action Button

    @ViewBuilder
    private var actionButton: some View {
        switch model.status {
        case .available:
            if model.sourceKind == .huggingFace {
                Button {
                    Task {
                        await modelManager.downloadModel(model)
                    }
                } label: {
                    Image(systemName: "arrow.down.circle")
                        .font(.title2)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.orange)
            } else {
                Label("Missing", systemImage: "exclamationmark.triangle")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }

        case .downloading:
            ProgressView()
                .controlSize(.small)

        case .downloaded:
            if modelManager.loadingModelId == model.id {
                // Loading in progress - show cancel button + share option
                HStack(spacing: 6) {
                    ModelLoadingIndicator()

                    Button {
                        showingCancelLoadAlert = true
                    } label: {
                        Image(systemName: "stop.circle.fill")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                    .controlSize(.small)

                    #if os(macOS)
                    if modelManager.sharingModelId == model.id {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Button {
                            Task {
                                await modelManager.shareModelToIOS(model)
                            }
                        } label: {
                            Image(systemName: "square.and.arrow.up")
                                .font(.caption)
                        }
                        .buttonStyle(.bordered)
                        .tint(.blue)
                        .controlSize(.small)
                        .disabled(modelManager.isSharingModel)
                    }
                    #endif
                }
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

                    #if os(macOS)
                    if modelManager.sharingModelId == model.id {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Button {
                            Task {
                                await modelManager.shareModelToIOS(model)
                            }
                        } label: {
                            Image(systemName: "square.and.arrow.up")
                                .font(.caption)
                        }
                        .buttonStyle(.bordered)
                        .tint(.blue)
                        .controlSize(.small)
                        .disabled(modelManager.isSharingModel)
                    }
                    #endif

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
                    .tint(isLoaded ? .green : .orange)
                    .controlSize(.small)
                    .disabled(isLoaded)
                }
            }

        case .error(let message):
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
                .help(message)
            } else {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .help(message)
            }
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
                Task {
                    await modelManager.shareModelToIOS(model)
                }
            } label: {
                Label("Share to iOS", systemImage: "square.and.arrow.up")
            }

            Divider()

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
            if model.sourceKind == .huggingFace {
                Button {
                    Task {
                        await modelManager.downloadModel(model)
                    }
                } label: {
                    Label("Download", systemImage: "arrow.down.circle")
                }
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
            if model.sourceKind == .huggingFace {
                Task {
                    await modelManager.downloadModel(model)
                }
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

/// Recommended sampling parameters from meta.yaml
struct RecommendedSampling: Sendable {
    let doSample: Bool
    let temperature: Double
    let topP: Double
    let topK: Int
}

/// Parsed model configuration from meta.yaml
struct ModelMetadata: Sendable {
    let version: String
    let modelType: String
    let architecture: String?
    let modelName: String?
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
    let recommendedSampling: RecommendedSampling?

    var isGemmaFamily: Bool {
        let fields = [
            modelPrefix.lowercased(),
            (architecture ?? "").lowercased(),
            (modelName ?? "").lowercased(),
            modelType.lowercased()
        ]
        return fields.contains { $0.contains("gemma") }
    }

    static func load(from path: String) -> ModelMetadata? {
        guard FileManager.default.fileExists(atPath: path),
              let content = try? String(contentsOfFile: path, encoding: .utf8) else {
            return nil
        }
        return loadFromString(content)
    }

    // [ANE-COMPAT:M1-A14] Parse from raw YAML string (used for pre-download compatibility check)
    static func loadFromString(_ content: String) -> ModelMetadata? {
        guard let yaml = try? Yams.load(yaml: content) as? [String: Any],
              let modelInfo = yaml["model_info"] as? [String: Any],
              let params = modelInfo["parameters"] as? [String: Any] else {
            return nil
        }

        // Parse recommended sampling if present
        var recommendedSampling: RecommendedSampling? = nil
        if let sampling = params["recommended_sampling"] as? [String: Any],
           let temperature = toDouble(sampling["temperature"]),
           let topP = toDouble(sampling["top_p"] ?? sampling["topP"]),
           let topK = toInt(sampling["top_k"] ?? sampling["topK"]) {
            let doSample = sampling["do_sample"] as? Bool ?? true
            recommendedSampling = RecommendedSampling(
                doSample: doSample,
                temperature: temperature,
                topP: topP,
                topK: topK
            )
        }

        return ModelMetadata(
            version: toString(modelInfo["version"]) ?? "Unknown",
            modelType: toString(modelInfo["model_type"]) ?? "chunked",
            architecture: toString(modelInfo["architecture"]),
            modelName: toString(modelInfo["name"]),
            modelPrefix: toString(params["model_prefix"]) ?? "unknown",
            contextLength: toInt(params["context_length"]) ?? 2048,
            batchSize: toInt(params["batch_size"]) ?? 32,
            lutFFN: toInt(params["lut_ffn"]),
            lutLMHead: toInt(params["lut_lmhead"]),
            lutEmbeddings: toInt(params["lut_embeddings"]),
            numChunks: toInt(params["num_chunks"]) ?? 1,
            splitLMHead: toInt(params["split_lm_head"]) ?? 8,
            argmaxInModel: toBool(params["argmax_in_model"]) ?? false,
            slidingWindow: toInt(params["sliding_window"]),
            recommendedSampling: recommendedSampling
        )
    }

    private static func toDouble(_ value: Any?) -> Double? {
        if let v = value as? Double { return v }
        if let v = value as? Float { return Double(v) }
        if let v = value as? Int { return Double(v) }
        if let v = value as? NSNumber { return v.doubleValue }
        if let v = value as? String { return Double(v) }
        return nil
    }

    private static func toInt(_ value: Any?) -> Int? {
        if let v = value as? Int { return v }
        if let v = value as? NSNumber { return v.intValue }
        if let v = value as? Double { return Int(v) }
        if let v = value as? Float { return Int(v) }
        if let v = value as? String { return Int(v) }
        return nil
    }

    private static func toBool(_ value: Any?) -> Bool? {
        if let v = value as? Bool { return v }
        if let v = value as? NSNumber { return v.boolValue }
        if let v = value as? String {
            switch v.lowercased() {
            case "true", "1", "yes", "y": return true
            case "false", "0", "no", "n": return false
            default: return nil
            }
        }
        return nil
    }

    private static func toString(_ value: Any?) -> String? {
        if let v = value as? String { return v }
        if let v = value as? NSNumber { return v.stringValue }
        return nil
    }
}

struct ModelDetailView: View {
    let model: ModelInfo
    @Environment(\.dismiss) private var dismiss
    @Environment(ModelManagerViewModel.self) private var modelManager

    @State private var metadata: ModelMetadata?
    @State private var isLoading = true
    @State private var weightDetails: (largest: Int64, largestName: String, files: [(name: String, size: Int64)])?
    @State private var weightWarning: String?
    @State private var computedDetailSize: String?

    private let accentGradient = LinearGradient(
        colors: [Color(red: 1.0, green: 0.6, blue: 0.2), Color(red: 1.0, green: 0.4, blue: 0.1)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                // Hero Header
                headerSection

                // Content Cards
                VStack(spacing: 16) {
                    // Quick Stats Bar
                    if let meta = metadata {
                        quickStatsBar(meta: meta)
                    }

                    // Main Info Card
                    infoCard

                    // Configuration Card (if metadata available)
                    if let meta = metadata {
                        configurationCard(meta: meta)
                        parametersCard(meta: meta)
                        quantizationCard(meta: meta)
                        // Sampling Card (if model has recommendations)
                        if let sampling = meta.recommendedSampling {
                            samplingCard(sampling: sampling, isArgmax: meta.argmaxInModel)
                        }
                    } else if model.isDownloaded && !isLoading {
                        noMetadataCard
                    } else if !model.isDownloaded {
                        downloadPromptCard
                    }

                    // Weight Files Card (for downloaded models)
                    if model.isDownloaded {
                        weightFilesCard
                    }

                    // Storage Card
                    if let path = model.localPath {
                        storageCard(path: path)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 24)
            }
        }
        .background(Color(white: 0.08))
        .overlay(alignment: .topTrailing) {
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .padding(16)
        }
        .task {
            await loadMetadata()
        }
    }

    // MARK: - Hero Header

    private var headerSection: some View {
        VStack(spacing: 16) {
            // Model Icon
            ZStack {
                Circle()
                    .fill(accentGradient.opacity(0.2))
                    .frame(width: 80, height: 80)

                Circle()
                    .strokeBorder(accentGradient, lineWidth: 2)
                    .frame(width: 80, height: 80)

                Image(systemName: architectureIcon)
                    .font(.system(size: 32, weight: .medium))
                    .foregroundStyle(accentGradient)
            }

            // Model Name
            Text(model.name)
                .font(.title2)
                .fontWeight(.bold)
                .multilineTextAlignment(.center)

            // Architecture Badge
            if let arch = model.architecture {
                Text(arch.uppercased())
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                    .background(accentGradient, in: Capsule())
            }

            // Status Badge
            statusBadge
        }
        .padding(.vertical, 32)
        .frame(maxWidth: .infinity)
        .background(
            LinearGradient(
                colors: [Color(white: 0.12), Color(white: 0.08)],
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }

    private var architectureIcon: String {
        switch model.architecture?.lowercased() {
        case "gemma": return "brain"
        case "llama": return "hare"
        case "qwen": return "sparkles"
        case "deepseek": return "waveform.path.ecg"
        default: return "cpu"
        }
    }

    private var statusBadge: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            Text(statusText)
                .font(.caption)
                .fontWeight(.medium)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(statusColor.opacity(0.15), in: Capsule())
        .foregroundStyle(statusColor)
    }

    private var statusColor: Color {
        switch model.status {
        case .available: return .orange
        case .downloading: return .yellow
        case .downloaded: return .green
        case .error: return .red
        }
    }

    private var statusText: String {
        switch model.status {
        case .available: return "Available"
        case .downloading: return "Downloading..."
        case .downloaded: return "Downloaded"
        case .error: return "Error"
        }
    }

    // MARK: - Quick Stats Bar

    private func quickStatsBar(meta: ModelMetadata) -> some View {
        HStack(spacing: 0) {
            quickStat(value: "\(meta.contextLength)", label: "Context", icon: "text.alignleft")
            Divider().frame(height: 40)
            quickStat(value: "\(meta.batchSize)", label: "Batch", icon: "square.grid.2x2")
            Divider().frame(height: 40)
            quickStat(value: computedDetailSize ?? model.size, label: "Size", icon: "externaldrive")
            Divider().frame(height: 40)
            quickStat(value: "\(meta.numChunks)", label: "Chunks", icon: "square.stack.3d.up")
        }
        .padding(.vertical, 12)
        .background(Color(white: 0.12), in: RoundedRectangle(cornerRadius: 12))
    }

    private func quickStat(value: String, label: String, icon: String) -> some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.body, design: .rounded))
                .fontWeight(.bold)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Info Card

    private var infoCard: some View {
        DetailCard(title: "Model Information", icon: "info.circle.fill", iconColor: .orange) {
            DetailCardRow(label: "Model ID", value: model.id, isMonospace: true)
            DetailCardRow(label: "Size", value: computedDetailSize ?? model.size)
            if let arch = model.architecture {
                DetailCardRow(label: "Architecture", value: arch.capitalized)
            }
        }
    }

    // MARK: - Configuration Card

    private func configurationCard(meta: ModelMetadata) -> some View {
        DetailCard(title: "Configuration", icon: "gearshape.fill", iconColor: Color(red: 1.0, green: 0.5, blue: 0.2)) {
            DetailCardRow(label: "Version", value: meta.version, badge: true)
            DetailCardRow(label: "Model Type", value: meta.modelType.capitalized)
            DetailCardRow(label: "Model Prefix", value: meta.modelPrefix, isMonospace: true)
        }
    }

    // MARK: - Parameters Card

    private func parametersCard(meta: ModelMetadata) -> some View {
        DetailCard(title: "Parameters", icon: "slider.horizontal.3", iconColor: .orange) {
            DetailCardRow(label: "Context Length", value: "\(meta.contextLength) tokens")
            DetailCardRow(label: "Batch Size", value: "\(meta.batchSize)")
            DetailCardRow(label: "Split LM Head", value: "\(meta.splitLMHead)")
            if let sw = meta.slidingWindow {
                DetailCardRow(label: "Sliding Window", value: "\(sw)")
            }
            DetailCardRow(label: "Argmax in Model", value: meta.argmaxInModel ? "Yes" : "No",
                         valueColor: meta.argmaxInModel ? .green : .secondary)
        }
    }

    // MARK: - Quantization Card

    private func quantizationCard(meta: ModelMetadata) -> some View {
        DetailCard(title: "Quantization", icon: "cube.fill", iconColor: Color(red: 1.0, green: 0.7, blue: 0.3)) {
            HStack(spacing: 12) {
                quantBadge(label: "FFN", bits: meta.lutFFN)
                quantBadge(label: "LM Head", bits: meta.lutLMHead)
                quantBadge(label: "Embed", bits: meta.lutEmbeddings)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
        }
    }

    private func quantBadge(label: String, bits: Int?) -> some View {
        VStack(spacing: 4) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)

            if let bits = bits, bits > 0 {
                Text("\(bits)-bit")
                    .font(.system(.caption, design: .rounded))
                    .fontWeight(.bold)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(quantColor(bits: bits), in: Capsule())
            } else {
                Text("FP16")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color(white: 0.2), in: Capsule())
            }
        }
        .frame(maxWidth: .infinity)
    }

    private func quantColor(bits: Int) -> Color {
        switch bits {
        case 1...4: return .orange
        case 5...6: return Color(red: 1.0, green: 0.5, blue: 0.2)
        case 7...8: return Color(red: 0.9, green: 0.4, blue: 0.1)
        default: return Color(red: 0.8, green: 0.3, blue: 0.1)
        }
    }

    // MARK: - Sampling Card

    private func samplingCard(sampling: RecommendedSampling, isArgmax: Bool) -> some View {
        DetailCard(title: "Recommended Sampling", icon: "dice.fill", iconColor: .green) {
            if isArgmax {
                // Argmax model - sampling not available
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text("Sampling unavailable (argmax model)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                HStack(spacing: 12) {
                    samplingBadge(label: "Temp", value: String(format: "%.2f", sampling.temperature))
                    samplingBadge(label: "Top-P", value: String(format: "%.2f", sampling.topP))
                    samplingBadge(label: "Top-K", value: sampling.topK == 0 ? "Off" : "\(sampling.topK)")
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)

                HStack(spacing: 4) {
                    Image(systemName: sampling.doSample ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundStyle(sampling.doSample ? .green : .secondary)
                    Text(sampling.doSample ? "Sampling enabled" : "Greedy decoding")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func samplingBadge(label: String, value: String) -> some View {
        VStack(spacing: 4) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)

            Text(value)
                .font(.system(.caption, design: .rounded))
                .fontWeight(.bold)
                .foregroundStyle(.white)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(Color.green, in: Capsule())
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Weight Files Card

    @ViewBuilder
    private var weightFilesCard: some View {
        DetailCard(title: "Weight Files", icon: "scalemass.fill", iconColor: Color(red: 0.6, green: 0.8, blue: 1.0)) {
            VStack(alignment: .leading, spacing: 12) {
                if let details = weightDetails {
                    // Largest weight file
                    DetailCardRow(
                        label: "Largest Weight",
                        value: formatBytes(details.largest),
                        valueColor: details.largest > DeviceType.maxWeightFileSize ? .red : .primary
                    )

                    DetailCardRow(
                        label: "From Model",
                        value: details.largestName,
                        isMonospace: true
                    )

                    DetailCardRow(
                        label: "Total Weight Files",
                        value: "\(details.files.count)"
                    )

                    // Warning if weight exceeds 1GB
                    if let warning = weightWarning {
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.white)
                                .font(.system(size: 14))

                            VStack(alignment: .leading, spacing: 2) {
                                Text("Device Compatibility Warning")
                                    .font(.caption)
                                    .fontWeight(.semibold)
                                    .foregroundStyle(.white)

                                Text(warning)
                                    .font(.caption2)
                                    .foregroundStyle(.white.opacity(0.9))
                                    .lineLimit(3)
                            }

                            Spacer()
                        }
                        .padding(10)
                        .background(Color.red.opacity(0.85), in: RoundedRectangle(cornerRadius: 8))
                    } else if details.largest > DeviceType.maxWeightFileSize {
                        // Show informational message for Mac users
                        HStack(spacing: 8) {
                            Image(systemName: "info.circle.fill")
                                .foregroundStyle(.orange)
                                .font(.system(size: 14))

                            Text("Weight file exceeds 1GB. This model may not load on iPhone or non-M-series iPad.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        .padding(8)
                        .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                    }
                } else if isLoading {
                    HStack {
                        ProgressView()
                            .controlSize(.small)
                        Text("Loading weight info...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text("No weight files found")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    // MARK: - Storage Card

    private func storageCard(path: String) -> some View {
        DetailCard(title: "Storage", icon: "folder.fill", iconColor: Color(red: 1.0, green: 0.6, blue: 0.15)) {
            VStack(alignment: .leading, spacing: 8) {
                Text(path)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .truncationMode(.middle)

                #if os(macOS)
                Button {
                    let url = URL(fileURLWithPath: path)
                    NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: url.path)
                } label: {
                    Label("Show in Finder", systemImage: "folder.badge.gearshape")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                #endif
            }
        }
    }

    // MARK: - Placeholder Cards

    private var noMetadataCard: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.title)
                .foregroundStyle(.orange)
            Text("Could not load model configuration")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
        .background(Color(white: 0.12), in: RoundedRectangle(cornerRadius: 12))
    }

    private var downloadPromptCard: some View {
        VStack(spacing: 12) {
            Image(systemName: "arrow.down.circle")
                .font(.title)
                .foregroundStyle(.orange)
            Text("Download the model to view detailed configuration")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
        .background(Color(white: 0.12), in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Load Metadata

    private func loadMetadata() async {
        isLoading = true
        defer { isLoading = false }

        guard model.isDownloaded else { return }

        // Use withLinkedModelAccess for security-scoped bookmark resolution.
        // getWeightFileDetails/formattedModelSize resolve bookmarks internally,
        // but ModelMetadata.load reads the file directly so we need access here.
        modelManager.withLinkedModelAccess(for: model) { modelURL in
            let metaPath = modelURL.appendingPathComponent("meta.yaml").path
            metadata = ModelMetadata.load(from: metaPath)
        }

        // Compute actual size for local models
        if model.size == "Local" {
            computedDetailSize = modelManager.formattedModelSize(for: model)
        }

        // Load weight file details (these resolve bookmarks internally)
        weightDetails = modelManager.getWeightFileDetails(for: model)
        weightWarning = modelManager.getWeightSizeWarning(for: model)
    }
}

// MARK: - Detail Card

private struct DetailCard<Content: View>: View {
    let title: String
    let icon: String
    let iconColor: Color
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.subheadline)
                    .foregroundStyle(iconColor)
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)
            }

            // Content
            content
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(white: 0.12), in: RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Detail Card Row

private struct DetailCardRow: View {
    let label: String
    let value: String
    var isMonospace: Bool = false
    var badge: Bool = false
    var valueColor: Color = .primary

    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Spacer()

            if badge {
                Text(value)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Color.orange, in: Capsule())
            } else {
                Text(value)
                    .font(isMonospace ? .system(.subheadline, design: .monospaced) : .subheadline)
                    .fontWeight(.medium)
                    .foregroundStyle(valueColor)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
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
