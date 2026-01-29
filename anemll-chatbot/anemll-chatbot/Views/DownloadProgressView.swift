// Copyright (c) 2025 Anemll
// Licensed under the MIT License
// DownloadProgressView.swift - Enhanced download progress indicator

import SwiftUI

/// A visually appealing download progress indicator
struct DownloadProgressView: View {
    let progress: Double
    let statusText: String
    let modelName: String

    // Parse the status text to extract components
    // New Format: "ModelName: 45% (156.2/355.9 MB) - ~5 min"
    // Old Format: "ModelName: filename - 45% (156.2/355.9 MB) - ~5 min"
    private var parsedStatus: (modelDisplayName: String, percent: Int, downloaded: String, total: String, eta: String) {
        // Default values
        var modelDisplayName = modelName
        var percent = Int(progress * 100)
        var downloaded = ""
        var total = ""
        var eta = ""

        // Try to parse the status text
        // New format example: "Qwen 0.3b: 45% (156.2/355.9 MB) - ~5 min"
        let parts = statusText.components(separatedBy: " - ")

        if parts.count >= 1 {
            // First part: "ModelName: 45% (156.2/355.9 MB)"
            let firstPart = parts[0]

            // Extract model name (before colon)
            if let colonIndex = firstPart.firstIndex(of: ":") {
                modelDisplayName = String(firstPart[..<colonIndex]).trimmingCharacters(in: .whitespaces)
            }

            // Extract percentage and bytes from first part (new format)
            if let percentRange = firstPart.range(of: #"\d+%"#, options: .regularExpression) {
                let percentStr = firstPart[percentRange].replacingOccurrences(of: "%", with: "")
                percent = Int(percentStr) ?? percent
            }

            // Extract bytes: (156.2/355.9 MB)
            if let openParen = firstPart.firstIndex(of: "("),
               let closeParen = firstPart.firstIndex(of: ")") {
                let bytesStr = String(firstPart[firstPart.index(after: openParen)..<closeParen])
                let byteParts = bytesStr.components(separatedBy: "/")
                if byteParts.count == 2 {
                    downloaded = byteParts[0].trimmingCharacters(in: .whitespaces)
                    total = byteParts[1].trimmingCharacters(in: .whitespaces)
                }
            }
        }

        // Check for ETA in second part (new format) or third part (old format)
        if parts.count >= 2 {
            // Could be ETA (new format) or progress part (old format)
            let secondPart = parts[1].trimmingCharacters(in: .whitespaces)

            // Check if this is an ETA string (starts with ~ or < or contains "min" or "hour")
            if secondPart.hasPrefix("~") || secondPart.hasPrefix("<") ||
               secondPart.contains("min") || secondPart.contains("hour") {
                eta = secondPart
            } else {
                // Old format: second part is progress, check for bytes here
                if let percentRange = secondPart.range(of: #"\d+%"#, options: .regularExpression) {
                    let percentStr = secondPart[percentRange].replacingOccurrences(of: "%", with: "")
                    percent = Int(percentStr) ?? percent
                }

                if let openParen = secondPart.firstIndex(of: "("),
                   let closeParen = secondPart.firstIndex(of: ")") {
                    let bytesStr = String(secondPart[secondPart.index(after: openParen)..<closeParen])
                    let byteParts = bytesStr.components(separatedBy: "/")
                    if byteParts.count == 2 {
                        downloaded = byteParts[0].trimmingCharacters(in: .whitespaces)
                        total = byteParts[1].trimmingCharacters(in: .whitespaces)
                    }
                }

                // Look for ETA in third part (old format)
                if parts.count >= 3 {
                    eta = parts[2].trimmingCharacters(in: .whitespaces)
                }
            }
        }

        return (modelDisplayName, percent, downloaded, total, eta)
    }

    private var hasError: Bool {
        statusText.lowercased().contains("error") ||
        statusText.lowercased().contains("failed") ||
        statusText.lowercased().contains("timeout")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if hasError {
                // Error state
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.red)
                        .font(.title3)

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Download Error")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(.red)

                        Text(statusText)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(2)
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.red.opacity(0.1))
                .cornerRadius(10)
            } else {
                // Normal download progress
                VStack(alignment: .leading, spacing: 10) {
                    // Top row: Model name and ETA
                    HStack {
                        // Download icon and model name
                        HStack(spacing: 6) {
                            Image(systemName: "arrow.down.circle.fill")
                                .foregroundColor(.blue)
                                .font(.system(size: 16))

                            Text(parsedStatus.modelDisplayName.isEmpty ? "Downloading..." : parsedStatus.modelDisplayName)
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundColor(.primary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }

                        Spacer()

                        // ETA badge
                        if !parsedStatus.eta.isEmpty {
                            Text(parsedStatus.eta)
                                .font(.caption)
                                .fontWeight(.medium)
                                .foregroundColor(.white)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.blue.opacity(0.8))
                                .cornerRadius(6)
                        }
                    }

                    // Custom progress bar
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            // Background
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.gray.opacity(0.2))
                                .frame(height: 12)

                            // Progress fill with gradient
                            RoundedRectangle(cornerRadius: 6)
                                .fill(
                                    LinearGradient(
                                        gradient: Gradient(colors: [Color.blue, Color.blue.opacity(0.7)]),
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: max(0, geometry.size.width * CGFloat(max(0.01, progress))), height: 12)
                                .animation(.easeInOut(duration: 0.3), value: progress)

                            // Percentage text overlay (centered in the bar)
                            Text("\(parsedStatus.percent)%")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundColor(.white)
                                .shadow(color: .black.opacity(0.3), radius: 1, x: 0, y: 1)
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .frame(height: 12)

                    // Bottom row: Downloaded / Total
                    HStack {
                        if !parsedStatus.downloaded.isEmpty && !parsedStatus.total.isEmpty {
                            HStack(spacing: 4) {
                                Text(parsedStatus.downloaded)
                                    .font(.caption)
                                    .fontWeight(.semibold)
                                    .foregroundColor(.primary)

                                Text("/")
                                    .font(.caption)
                                    .foregroundColor(.secondary)

                                Text(parsedStatus.total)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        } else {
                            Text("Starting download...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        // Animated activity indicator
                        ProgressView()
                            .scaleEffect(0.7)
                    }
                }
                .padding(12)
                .background(Color.blue.opacity(0.05))
                .cornerRadius(10)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.blue.opacity(0.2), lineWidth: 1)
                )
            }
        }
        .padding(.top, 4)
    }
}

// Preview
struct DownloadProgressView_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 20) {
            // New format: total bytes progress without individual file names
            DownloadProgressView(
                progress: 0.45,
                statusText: "Qwen 0.3b: 45% (156.2/355.9 MB) - ~5 min",
                modelName: "Qwen 0.3b"
            )

            DownloadProgressView(
                progress: 0.02,
                statusText: "Starting download...",
                modelName: "Llama 3.2"
            )

            DownloadProgressView(
                progress: 0.0,
                statusText: "Error: Connection timed out",
                modelName: "Test Model"
            )

            // Test with calculating state
            DownloadProgressView(
                progress: 0.05,
                statusText: "Calculating total download size...",
                modelName: "Gemma 3"
            )
        }
        .padding()
    }
}
