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

// MARK: - HuggingFace Collection Factory

extension ModelInfo {
    /// Create a ModelInfo from a HuggingFace repo ID and fetched metadata
    static func fromHuggingFaceRepo(id: String, sizeBytes: Int64?, modelType: String?) -> ModelInfo {
        // Strip "anemll/anemll-" prefix to get the model descriptor
        let repoName = id.components(separatedBy: "/").last ?? id
        let descriptor = repoName.hasPrefix("anemll-") ? String(repoName.dropFirst(7)) : repoName

        // Parse context length from "ctx\d+" pattern
        let contextLength: Int? = {
            guard let range = descriptor.range(of: #"ctx(\d+)"#, options: .regularExpression) else { return nil }
            let match = descriptor[range]
            let digits = match.dropFirst(3) // drop "ctx"
            return Int(digits)
        }()

        // Detect architecture from modelType or repo name
        let arch = modelType ?? detectArchitecture(from: descriptor)

        // Build friendly display name
        let name = buildDisplayName(from: descriptor)

        // Format size
        let sizeString: String
        if let bytes = sizeBytes {
            sizeString = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
        } else {
            sizeString = "Unknown"
        }

        // Build description
        let desc = buildDescription(from: descriptor, architecture: arch, contextLength: contextLength)

        return ModelInfo(
            id: id,
            name: name,
            description: desc,
            size: sizeString,
            sizeBytes: sizeBytes,
            contextLength: contextLength,
            architecture: arch,
            sourceKind: .huggingFace
        )
    }

    private static func detectArchitecture(from descriptor: String) -> String? {
        let lower = descriptor.lowercased()
        if lower.contains("gemma") { return "gemma" }
        if lower.contains("qwen") { return "qwen" }
        if lower.contains("llama") { return "llama" }
        if lower.contains("deepseek") { return "deepseek" }
        if lower.contains("deephermes") { return "deephermes" }
        return nil
    }

    private static func buildDisplayName(from descriptor: String) -> String {
        let lower = descriptor.lowercased()

        // Extract parameter size (e.g., "270m", "1b", "1.7B", "0.5B", "4B", "8B")
        let paramSize: String? = {
            // Match patterns like "3-270m", "3-1b", "1.7B", "0.5B" etc.
            if let range = descriptor.range(of: #"[\d.]+-?(\d+\.?\d*[BbMm])"#, options: .regularExpression) {
                // Extract just the size part
                let match = String(descriptor[range])
                // Find the actual size at the end
                if let sizeRange = match.range(of: #"\d+\.?\d*[BbMm]$"#, options: .regularExpression) {
                    return String(match[sizeRange]).uppercased()
                }
            }
            // Try standalone size like "1.7B" or "0.5B"
            if let range = descriptor.range(of: #"(\d+\.?\d*[BbMm])\b"#, options: .regularExpression) {
                return String(descriptor[range]).uppercased()
            }
            return nil
        }()

        // Determine model family
        if lower.contains("gemma") {
            let version = lower.contains("gemma-3") || lower.contains("gemma3") ? "3" : ""
            return "Gemma \(version) \(paramSize ?? "")".trimmingCharacters(in: .whitespaces)
        }
        if lower.contains("qwen3") {
            return "Qwen3 \(paramSize ?? "")".trimmingCharacters(in: .whitespaces)
        }
        if lower.contains("qwen2.5") || lower.contains("qwen-2.5") {
            return "Qwen 2.5 \(paramSize ?? "")".trimmingCharacters(in: .whitespaces)
        }
        if lower.contains("qwen") {
            return "Qwen \(paramSize ?? "")".trimmingCharacters(in: .whitespaces)
        }
        if lower.contains("llama") {
            let version = lower.contains("3.2") ? "3.2" : lower.contains("3.1") ? "3.1" : ""
            return "LLaMA \(version) \(paramSize ?? "")".trimmingCharacters(in: .whitespaces)
        }
        if lower.contains("deephermes") {
            return "DeepHermes \(paramSize ?? "")".trimmingCharacters(in: .whitespaces)
        }
        if lower.contains("deepseek") {
            return "DeepSeek \(paramSize ?? "")".trimmingCharacters(in: .whitespaces)
        }

        // Fallback: use descriptor cleaned up
        return descriptor
            .replacingOccurrences(of: #"_\d+\.\d+\.\d+$"#, with: "", options: .regularExpression)
            .replacingOccurrences(of: "-", with: " ")
    }

    private static func buildDescription(from descriptor: String, architecture: String?, contextLength: Int?) -> String {
        var parts: [String] = []

        if let arch = architecture {
            let source: String
            switch arch.lowercased() {
            case "gemma": source = "Google"
            case "qwen": source = "Alibaba"
            case "llama": source = "Meta"
            case "deepseek": source = "DeepSeek"
            case "deephermes": source = "Nous Research"
            default: source = ""
            }
            if !source.isEmpty { parts.append(source) }
        }

        if descriptor.lowercased().contains("monolithic") {
            parts.append("monolithic")
        }

        if let ctx = contextLength {
            if ctx >= 1024 {
                parts.append("\(ctx / 1024)K context")
            } else {
                parts.append("\(ctx) context")
            }
        }

        if parts.isEmpty { return "ANEMLL optimized model" }
        return parts.joined(separator: " · ")
    }
}

// MARK: - Default Models

extension ModelInfo {
    /// Default available models from HuggingFace (fallback when collection fetch fails)
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
            id: "anemll/anemll-google-gemma-3-270m-it-ctx4096_0.3.5",
            name: "Gemma 3 1B",
            description: "Google's Gemma 3 1B with 4K context",
            size: "1.5 GB",
            sizeBytes: 1_600_000_000,
            contextLength: 4096,
            architecture: "gemma"
        )
    ]
}
