// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// ModelListView.swift

import SwiftUI

struct ModelListView: View {
    let model: Model
    let modelService: ModelService
    let inferenceService: InferenceService
    let onSelect: () -> Void
    let onDownload: () -> Void
    @State private var isDownloading = false
    @State private var downloadProgress: Double = 0.0
    @State private var currentFile: String = ""
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(model.name)
                    .font(.headline)
                
                Spacer()
                
                // Show model status
                if modelService.isModelSelected(model) && inferenceService.isModelLoaded && inferenceService.currentModel == model.id {
                    Label("Loaded", systemImage: "checkmark.circle.fill")
                        .foregroundColor(.green)
                        .font(.caption)
                } else if modelService.isModelSelected(model) && inferenceService.isLoadingModel {
                    HStack(spacing: 4) {
                        ProgressView()
                            .scaleEffect(0.7)
                        Text("\(Int(inferenceService.loadingProgress * 100))%")
                            .font(.caption)
                            .foregroundColor(.blue)
                    }
                } else if model.isDownloaded {
                    Text("Downloaded")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else if let progress = modelService.getDownloadProgress(for: model.id) {
                    Text("\(Int(progress * 100))%")
                        .font(.caption)
                        .foregroundColor(.blue)
                } else {
                    Text("Not Downloaded")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            // Action buttons
            HStack {
                // Download button
                if !model.isDownloaded {
                    Button(action: {
                        isDownloading = true
                        modelService.downloadModel(modelId: model.id, fileProgress: { file, progress in
                            downloadProgress = progress
                            currentFile = file
                        }) { success in
                            isDownloading = false
                            if success {
                                // Automatically load the model after download
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                    modelService.loadModel(model)
                                }
                            }
                        }
                    }) {
                        Text("Download")
                            .fontWeight(.medium)
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.blue)
                            .cornerRadius(8)
                    }
                    .disabled(isDownloading)
                } else {
                    // Load button
                    Button(action: {
                        modelService.loadModel(model)
                        dismiss()
                    }) {
                        Text("Load")
                            .fontWeight(.medium)
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(modelService.isModelSelected(model) && inferenceService.isModelLoaded && inferenceService.currentModel == model.id ? Color.gray : Color.blue)
                            .cornerRadius(8)
                    }
                    .disabled(modelService.isModelSelected(model) && inferenceService.isModelLoaded && inferenceService.currentModel == model.id || inferenceService.isLoadingModel)
                }
                
                Spacer()
            }
            .padding(.top, 4)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(10)
    }
}

#Preview {
    ModelListView(
        model: Model(id: "test-model", name: "Test Model", description: "A test model", size: 1000000, downloadURL: ""),
        modelService: ModelService.shared,
        inferenceService: InferenceService.shared,
        onSelect: {},
        onDownload: {}
    )
} 