//
//  ModelInfo.swift
//  ANEMLLChat
//
//  Model metadata and download state
//

import Foundation

/// Model source type for lifecycle handling (download/import/link)
enum ModelSourceKind: String, Codable, Sendable {
    case huggingFace
    case localImported
    case localLinked
}

/// Information about an available LLM model
struct ModelInfo: Identifiable, Codable, Sendable, Equatable {
    let id: String                    // HuggingFace repo ID (e.g., "anemll/llama-3.2-1B")
    let name: String                  // Display name
    let description: String           // Model description
    let size: String                  // Human-readable size (e.g., "1.2 GB")
    let sizeBytes: Int64?             // Size in bytes for calculations

    // Model capabilities
    let contextLength: Int?
    let architecture: String?         // llama, qwen, gemma, deepseek

    // Download state
    var isDownloaded: Bool
    var downloadProgress: Double?
    var downloadedBytes: Int64?
    var downloadError: String?
    var isDownloading: Bool

    // Local paths (set after download)
    var localPath: String?
    var metaYamlPath: String?
    var sourceKind: ModelSourceKind
    var linkedPath: String?
    var bookmarkDataBase64: String?

    init(
        id: String,
        name: String,
        description: String = "",
        size: String = "Unknown",
        sizeBytes: Int64? = nil,
        contextLength: Int? = nil,
        architecture: String? = nil,
        isDownloaded: Bool = false,
        downloadProgress: Double? = nil,
        downloadedBytes: Int64? = nil,
        downloadError: String? = nil,
        isDownloading: Bool = false,
        localPath: String? = nil,
        metaYamlPath: String? = nil,
        sourceKind: ModelSourceKind = .huggingFace,
        linkedPath: String? = nil,
        bookmarkDataBase64: String? = nil
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.size = size
        self.sizeBytes = sizeBytes
        self.contextLength = contextLength
        self.architecture = architecture
        self.isDownloaded = isDownloaded
        self.downloadProgress = downloadProgress
        self.downloadedBytes = downloadedBytes
        self.downloadError = downloadError
        self.isDownloading = isDownloading
        self.localPath = localPath
        self.metaYamlPath = metaYamlPath
        self.sourceKind = sourceKind
        self.linkedPath = linkedPath
        self.bookmarkDataBase64 = bookmarkDataBase64
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case description
        case size
        case sizeBytes
        case contextLength
        case architecture
        case isDownloaded
        case downloadProgress
        case downloadedBytes
        case downloadError
        case isDownloading
        case localPath
        case metaYamlPath
        case sourceKind
        case linkedPath
        case bookmarkDataBase64
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(String.self, forKey: .id)
        self.name = try container.decode(String.self, forKey: .name)
        self.description = try container.decodeIfPresent(String.self, forKey: .description) ?? ""
        self.size = try container.decodeIfPresent(String.self, forKey: .size) ?? "Unknown"
        self.sizeBytes = try container.decodeIfPresent(Int64.self, forKey: .sizeBytes)
        self.contextLength = try container.decodeIfPresent(Int.self, forKey: .contextLength)
        self.architecture = try container.decodeIfPresent(String.self, forKey: .architecture)
        self.isDownloaded = try container.decodeIfPresent(Bool.self, forKey: .isDownloaded) ?? false
        self.downloadProgress = try container.decodeIfPresent(Double.self, forKey: .downloadProgress)
        self.downloadedBytes = try container.decodeIfPresent(Int64.self, forKey: .downloadedBytes)
        self.downloadError = try container.decodeIfPresent(String.self, forKey: .downloadError)
        self.isDownloading = try container.decodeIfPresent(Bool.self, forKey: .isDownloading) ?? false
        self.localPath = try container.decodeIfPresent(String.self, forKey: .localPath)
        self.metaYamlPath = try container.decodeIfPresent(String.self, forKey: .metaYamlPath)
        self.sourceKind = try container.decodeIfPresent(ModelSourceKind.self, forKey: .sourceKind) ?? .huggingFace
        self.linkedPath = try container.decodeIfPresent(String.self, forKey: .linkedPath)
        self.bookmarkDataBase64 = try container.decodeIfPresent(String.self, forKey: .bookmarkDataBase64)
    }
}

// MARK: - Download Progress

extension ModelInfo {
    /// Formatted download progress string
    var downloadProgressString: String? {
        guard let progress = downloadProgress else { return nil }
        return String(format: "%.0f%%", progress * 100)
    }

    /// Formatted bytes downloaded
    var downloadedBytesString: String? {
        guard let bytes = downloadedBytes else { return nil }
        return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    /// ETA calculation based on download speed
    func estimatedTimeRemaining(bytesPerSecond: Double) -> String? {
        guard let total = sizeBytes,
              let downloaded = downloadedBytes,
              bytesPerSecond > 0 else { return nil }

        let remaining = Double(total - downloaded)
        let seconds = remaining / bytesPerSecond

        if seconds < 60 {
            return String(format: "%.0fs remaining", seconds)
        } else if seconds < 3600 {
            return String(format: "%.0fm remaining", seconds / 60)
        } else {
            return String(format: "%.1fh remaining", seconds / 3600)
        }
    }
}

// MARK: - Model Status

extension ModelInfo {
    enum Status: Equatable {
        case available          // Not downloaded, available to download
        case downloading        // Currently downloading
        case downloaded         // Downloaded and ready
        case error(String)      // Download failed
    }

    var status: Status {
        if let error = downloadError {
            return .error(error)
        }
        if isDownloading {
            return .downloading
        }
        if isDownloaded {
            return .downloaded
        }
        return .available
    }

    var statusIcon: String {
        switch status {
        case .available: return "cloud"
        case .downloading: return "arrow.down.circle.fill"
        case .downloaded: return "checkmark.circle.fill"
        case .error: return "exclamationmark.triangle.fill"
        }
    }
}

// MARK: - Default Models

extension ModelInfo {
    /// Default available models from HuggingFace
    /// Gemma 3 270M first (smallest/fastest), then 1B, then others
    static let defaultModels: [ModelInfo] = [
        ModelInfo(
            id: "anemll/anemll-google-gemma-3-270m-it-ctx512-monolithic_0.3.5",
            name: "Gemma 3 270M",
            description: "Google's Gemma 3 270M - fast & compact",
            size: "0.5 GB",
            sizeBytes: 500_000_000,
            contextLength: 512,
            architecture: "gemma"
        ),
        ModelInfo(
            id: "anemll/anemll-google-gemma-3-1b-it-ctx4096_0.3.4",
            name: "Gemma 3 1B",
            description: "Google's Gemma 3 1B with 4K context",
            size: "1.5 GB",
            sizeBytes: 1_600_000_000,
            contextLength: 4096,
            architecture: "gemma"
        ),
        ModelInfo(
            id: "anemll/anemll-llama-3.2-1B-iOSv2.0",
            name: "LLaMA 3.2 1B",
            description: "Meta's LLaMA 3.2 1B optimized for iOS/macOS",
            size: "1.6 GB",
            sizeBytes: 1_740_000_000,
            contextLength: 512,
            architecture: "llama"
        ),
        ModelInfo(
            id: "anemll/anemll-Qwen3-4B-ctx1024_0.3.0",
            name: "Qwen 3 4B",
            description: "Alibaba's Qwen 3 4B model",
            size: "4.0 GB",
            sizeBytes: 4_300_000_000,
            contextLength: 1024,
            architecture: "qwen"
        ),
        ModelInfo(
            id: "anemll/anemll-Llama-3.2-1B-FAST-iOS_0.3.0",
            name: "LLaMA 3.2 1B FAST",
            description: "Fast optimized LLaMA 3.2 1B",
            size: "1.2 GB",
            sizeBytes: 1_200_000_000,
            contextLength: 512,
            architecture: "llama"
        )
    ]
}
