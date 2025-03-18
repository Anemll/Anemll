import SwiftUI

// Component for a single model in the list with horizontal buttons
struct ModelListItemHorizontal: View {
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
    
    // Constants for uniform button sizing
    private let buttonWidth: CGFloat = 80
    private let buttonHeight: CGFloat = 36
    private let buttonCornerRadius: CGFloat = 8
    
    // Format file size nicely
    private func formatFileSize(_ size: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(size))
    }
    
    // Shared button style for consistent appearance
    private func ActionButton(title: String, color: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 13, weight: .medium))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .frame(width: buttonWidth, height: buttonHeight)
                .background(color)
                .foregroundColor(.white)
                .cornerRadius(buttonCornerRadius)
        }
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
            
            // Horizontal button row with uniform buttons
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    // Select button (always shown)
                    ActionButton(
                        title: isSelected ? "Selected" : "Select",
                        color: isSelected ? Color.green : Color.blue,
                        action: onSelect
                    )
                    
                    // Load button (only for downloaded models)
                    if isDownloaded && !isDownloading {
                        ActionButton(
                            title: "Load",
                            color: Color.purple,
                            action: onLoad
                        )
                    }
                    
                    // Download button (for downloaded models to verify/update)
                    if isDownloaded && !isDownloading {
                        ActionButton(
                            title: "Download",
                            color: hasIncompleteFiles ? Color.orange : Color.blue.opacity(0.8),
                            action: onDownload
                        )
                        .help("Download missing files or verify completeness")
                    }
                    
                    // Delete button (only for downloaded models)
                    if isDownloaded && !isDownloading {
                        ActionButton(
                            title: "Delete",
                            color: Color.red,
                            action: onDelete
                        )
                    }
                    
                    // Download button (for non-downloaded models)
                    if !isDownloaded && !isDownloading {
                        ActionButton(
                            title: "Download",
                            color: Color.blue,
                            action: onDownload
                        )
                    }
                    
                    // Cancel button (only for downloading models)
                    if isDownloading {
                        ActionButton(
                            title: "Cancel",
                            color: Color.red,
                            action: onCancelDownload
                        )
                    }
                }
                .padding(.vertical, 4)
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
