//
//  DownloadProgressView.swift
//  ANEMLLChat
//
//  Download progress indicator
//

import SwiftUI

struct DownloadProgressView: View {
    let progress: DownloadProgress

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Progress bar with percentage
            HStack(spacing: 8) {
                ProgressView(value: progress.progress)
                    .progressViewStyle(.linear)

                Text(String(format: "%.0f%%", progress.progress * 100))
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                    .frame(width: 36, alignment: .trailing)
            }

            // Stats row - simplified (no individual file names)
            HStack {
                // Total downloaded
                Text(progress.downloadedString)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                // Speed
                Text(progress.speedString)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                // ETA
                if let eta = progress.etaString {
                    Text("• \(eta)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}

// MARK: - Compact Progress View

struct CompactDownloadProgress: View {
    let progress: Double
    let speed: String?

    var body: some View {
        HStack(spacing: 8) {
            ProgressView(value: progress)
                .progressViewStyle(.linear)
                .frame(maxWidth: 100)

            Text(String(format: "%.0f%%", progress * 100))
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()

            if let speed = speed {
                Text(speed)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }
}

// MARK: - Circular Progress

struct CircularDownloadProgress: View {
    let progress: Double

    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.secondary.opacity(0.2), lineWidth: 3)

            Circle()
                .trim(from: 0, to: progress)
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.linear, value: progress)

            Text(String(format: "%.0f%%", progress * 100))
                .font(.caption2)
                .fontWeight(.medium)
        }
        .frame(width: 40, height: 40)
    }
}

#Preview {
    VStack(spacing: 24) {
        DownloadProgressView(progress: DownloadProgress(
            totalBytes: 1_000_000_000,
            downloadedBytes: 350_000_000,
            currentFile: "",
            filesCompleted: 2,
            totalFiles: 8,
            bytesPerSecond: 15_000_000
        ))

        CompactDownloadProgress(progress: 0.35, speed: "15 MB/s")

        CircularDownloadProgress(progress: 0.65)
    }
    .padding()
}
