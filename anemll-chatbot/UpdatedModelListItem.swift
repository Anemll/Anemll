// Updated ModelListItem with horizontal buttons
import SwiftUI

// Helper component for consistent button styling
struct ActionButton: View {
    let title: String
    let backgroundColor: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline)
                .fontWeight(.medium)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .frame(minWidth: 70)
                .background(backgroundColor)
                .foregroundColor(.white)
                .cornerRadius(8)
        }
    }
}

// Component for a single model in the list
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
            
            // Action buttons section - displayed horizontally
            HStack(spacing: 12) {
                // Select button
                ActionButton(
                    title: isSelected ? "Selected" : "Select",
                    backgroundColor: isSelected ? Color.green : Color.blue,
                    action: onSelect
                )
                
                // For downloaded models that aren't currently downloading
                if isDownloaded && !isDownloading {
                    // Load button
                    ActionButton(
                        title: "Load",
                        backgroundColor: Color.purple,
                        action: onLoad
                    )
                    
                    // Download button (renamed from Verify)
                    ActionButton(
                        title: "Download",
                        backgroundColor: hasIncompleteFiles ? Color.orange : Color.blue.opacity(0.8),
                        action: onDownload
                    )
                    .help("Download missing files or verify completeness")
                    
                    // Delete button
                    ActionButton(
                        title: "Delete",
                        backgroundColor: Color.red,
                        action: onDelete
                    )
                }
                
                // For models that aren't downloaded and aren't currently downloading
                if !isDownloaded && !isDownloading {
                    ActionButton(
                        title: "Download",
                        backgroundColor: Color.blue,
                        action: onDownload
                    )
                }
                
                // Cancel download button (only shown for downloading models)
                if isDownloading {
                    ActionButton(
                        title: "Cancel",
                        backgroundColor: Color.red,
                        action: onCancelDownload
                    )
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
