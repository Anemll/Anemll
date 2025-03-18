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
                    
                    HStack {
                        Text("Size: \(formatFileSize(model.size))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            
                        if isDownloaded && hasIncompleteFiles {
                            Text("Files incomplete")
                                .font(.caption)
                                .foregroundColor(.orange)
                        }
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    onShowInfo()
                }
                
                Spacer()
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
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Downloading...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        Text("\(Int(downloadProgress * 100))%")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    // Always use a non-zero progress value to ensure visibility
                    ProgressView(value: max(0.01, downloadProgress))
                        .progressViewStyle(LinearProgressViewStyle())
                        .animation(.easeInOut, value: downloadProgress)
                    
                    if !currentFile.isEmpty {
                        Text("Current file: \(currentFile)")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                .padding(.vertical, 4)
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
