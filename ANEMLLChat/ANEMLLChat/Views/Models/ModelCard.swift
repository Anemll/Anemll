//
//  ModelCard.swift
//  ANEMLLChat
//
//  Individual model card display
//

import SwiftUI

struct ModelCard: View {
    let model: ModelInfo

    @Environment(ModelManagerViewModel.self) private var modelManager

    @State private var showingDeleteAlert = false

    private var isLoaded: Bool {
        modelManager.loadedModelId == model.id
    }

    var body: some View {
        HStack(spacing: 12) {
            // Status icon
            statusIcon

            // Model info
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(model.name)
                        .font(.headline)

                    if isLoaded {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .font(.caption)
                    }
                }

                Text(model.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)

                HStack(spacing: 8) {
                    Label(model.size, systemImage: "internaldrive")
                        .font(.caption2)

                    if let context = model.contextLength {
                        Label("\(context) ctx", systemImage: "text.alignleft")
                            .font(.caption2)
                    }

                    if let arch = model.architecture {
                        Text(arch)
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.secondary.opacity(0.2), in: Capsule())
                    }
                }
                .foregroundStyle(.secondary)
            }

            Spacer()

            // Action button
            actionButton
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture {
            handleTap()
        }
        .contextMenu {
            contextMenuItems
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
                ProgressView()
                    .controlSize(.small)
            } else {
                Button {
                    Task {
                        await modelManager.loadModelForInference(model)
                    }
                } label: {
                    Text(isLoaded ? "Loaded" : "Load")
                        .font(.caption)
                        .fontWeight(.medium)
                }
                .buttonStyle(.borderedProminent)
                .tint(isLoaded ? .green : .blue)
                .disabled(isLoaded || modelManager.loadingModelId != nil)
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
