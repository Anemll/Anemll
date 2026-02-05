import SwiftUI

// Component for a single model in the list with horizontal buttons
struct ModelListItem: View {
    let model: Model
    let isDownloaded: Bool
    let isDownloading: Bool
    let downloadProgress: Double
    let currentFile: String
    let isSelected: Bool
    let onSelect: () -> Void
    let onLoad: () -> Void
    let onDelete: () -> Void
    let onDownload: () -> Void
    let onCancelDownload: () -> Void
    let onShowInfo: () -> Void
    
    // Check if the model has incomplete files
    let hasIncompleteFiles: Bool
    // Error message for incomplete/corrupt files
    var errorMessage: String? = nil
    // Warning message for weight file size issues (>1GB on iPhone/Bionic iPad)
    var weightSizeWarning: String? = nil

    // Format file size nicely
    private func formatFileSize(_ size: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(size))
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Model info section
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(model.name)
                            .font(.headline)
                            .foregroundColor(.primary)
                            
                        if isDownloaded && hasIncompleteFiles {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.orange)
                                .font(.caption)
                        }
                    }
                    
                    if !model.description.isEmpty {
                        Text(model.description)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Text("Size: \(formatFileSize(model.size))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    onShowInfo()
                }

                Spacer()
            }

            // Error banner for incomplete files - more visible
            if isDownloaded && hasIncompleteFiles {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.white)
                        .font(.system(size: 14))

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Download Required")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)

                        Text(errorMessage ?? "Model has missing or empty weight files")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.9))
                            .lineLimit(2)
                    }

                    Spacer()
                }
                .padding(10)
                .background(Color.orange)
                .cornerRadius(8)
            }

            // Warning banner for weight file size issues on iPhone/Bionic iPad
            if isDownloaded && !hasIncompleteFiles, let warning = weightSizeWarning {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.white)
                        .font(.system(size: 14))

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Device Compatibility Warning")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)

                        Text(warning)
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.9))
                            .lineLimit(3)
                    }

                    Spacer()
                }
                .padding(10)
                .background(Color.red.opacity(0.85))
                .cornerRadius(8)
            }
            
            // Horizontal button row
            HStack(spacing: 8) {
                // Select button (always shown)
                Button(action: onSelect) {
                    Text(isSelected ? "Selected" : "Select")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .padding(.vertical, 8)
                        .frame(maxWidth: .infinity)
                        .background(isSelected ? Color.green : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                
                // Load button (only for downloaded models)
                if isDownloaded && !isDownloading {
                    Button(action: onLoad) {
                        Text("Load")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .padding(.vertical, 8)
                            .frame(maxWidth: .infinity)
                            .background(Color.purple)
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                }
                
                // Download button (for downloaded models to verify/update)
                if isDownloaded && !isDownloading {
                    Button(action: onDownload) {
                        Text("Download")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .padding(.vertical, 8)
                            .frame(maxWidth: .infinity) 
                            .background(hasIncompleteFiles ? Color.orange : Color.blue.opacity(0.8))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                    .help("Download missing files or verify completeness")
                }
                
                // Delete button (only for downloaded models)
                if isDownloaded && !isDownloading {
                    Button(action: onDelete) {
                        Text("Delete")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .padding(.vertical, 8)
                            .frame(maxWidth: .infinity)
                            .background(Color.red)
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                }
                
                // Download button (for non-downloaded models)
                if !isDownloaded && !isDownloading {
                    Button(action: onDownload) {
                        Text("Download")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .padding(.vertical, 8)
                            .frame(maxWidth: .infinity)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                }
                
                // Cancel button (only for downloading models)
                if isDownloading {
                    Button(action: onCancelDownload) {
                        Text("Cancel")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .padding(.vertical, 8)
                            .frame(maxWidth: .infinity)
                            .background(Color.red)
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                }
            }
            
            // Download progress indicator (only shown when downloading)
            if isDownloading {
                DownloadProgressView(
                    progress: downloadProgress,
                    statusText: currentFile,
                    modelName: model.name
                )
            }
        }
        .padding()
        .background(isDownloaded && hasIncompleteFiles ? 
                    Color(.systemOrange).opacity(0.1) : 
                    Color(.secondarySystemBackground))
        .cornerRadius(10)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(isDownloaded && hasIncompleteFiles ? Color.orange.opacity(0.5) : Color.clear, lineWidth: 1)
        )
    }
}
