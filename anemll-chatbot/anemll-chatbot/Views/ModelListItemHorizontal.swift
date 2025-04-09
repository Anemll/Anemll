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
    // Add error message property
    let errorMessage: String?
    
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
                                .help("Model has missing or incomplete files")
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
                            HStack(spacing: 4) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundColor(.orange)
                                    .font(.caption)
                                if let error = errorMessage {
                                    Text(error)
                                        .font(.caption)
                                        .foregroundColor(.orange)
                                } else {
                                    Text("Missing required files")
                                        .font(.caption)
                                        .foregroundColor(.orange)
                                }
                            }
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
            HStack(spacing: 8) {
                // Select button (always shown)
                ActionButton(
                    title: isSelected ? "Selected" : "Select",
                    color: isSelected ? .green : .blue,
                    action: onSelect
                )
                
                // Load button (only for downloaded models)
                if isDownloaded && !isDownloading {
                    ActionButton(
                        title: "Load",
                        color: .purple,
                        action: onLoad
                    )
                }
                
                // Download/Verify button (for downloaded models)
                if isDownloaded && !isDownloading {
                    ActionButton(
                        title: "Download",
                        color: hasIncompleteFiles ? .orange : .blue.opacity(0.8),
                        action: onDownload
                    )
                    .help(hasIncompleteFiles ? "Download missing files" : "Verify model files")
                }
                
                // Delete button (only for downloaded models)
                if isDownloaded && !isDownloading {
                    ActionButton(
                        title: "Delete",
                        color: .red,
                        action: onDelete
                    )
                }
                
                // Download button (for non-downloaded models)
                if !isDownloaded && !isDownloading {
                    ActionButton(
                        title: "Download",
                        color: .blue,
                        action: onDownload
                    )
                }
                
                // Cancel button (only for downloading models)
                if isDownloading {
                    ActionButton(
                        title: "Cancel",
                        color: .red,
                        action: onCancelDownload
                    )
                }
            }
            
            // Download progress (only shown when downloading)
            if isDownloading {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: downloadProgress)
                        .progressViewStyle(LinearProgressViewStyle())
                    Text(currentFile)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(10)
    }
}
