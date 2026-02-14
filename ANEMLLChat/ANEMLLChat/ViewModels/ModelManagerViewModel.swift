//
//  ModelManagerViewModel.swift
//  ANEMLLChat
//
//  ViewModel for model management
//

import Foundation
import SwiftUI
import Observation
import CryptoKit
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

// MARK: - Device Type Detection

/// Device type for determining weight file size limits
enum DeviceType {
    case mac
    case macCatalyst
    case iPad
    case iPhone
    case visionPro
    case other

    static var current: DeviceType {
        #if os(macOS)
        return .mac
        #elseif targetEnvironment(macCatalyst)
        return .macCatalyst
        #elseif os(visionOS)
        return .visionPro
        #else
        // Check if running as iPad app on Vision Pro
        if isRunningOnVisionPro {
            return .visionPro
        }
        let device = UIDevice.current
        if device.userInterfaceIdiom == .pad {
            return .iPad
        } else if device.userInterfaceIdiom == .phone {
            return .iPhone
        } else {
            return .other
        }
        #endif
    }

    /// Detect if an iOS/iPadOS app is running on Apple Vision Pro (iPad compatibility mode)
    static var isRunningOnVisionPro: Bool {
        #if os(visionOS)
        return true
        #elseif os(iOS)
        // Use responds(to:) to safely check for isiOSAppOnVision —
        // direct property access crashes on older OS versions where the selector doesn't exist
        let pi = ProcessInfo.processInfo
        if pi.responds(to: NSSelectorFromString("isiOSAppOnVision")),
           let value = pi.value(forKey: "isiOSAppOnVision") as? Bool {
            return value
        }
        // Fallback: check if machine identifier indicates a RealityDevice
        let machine = machineIdentifier
        if machine.hasPrefix("RealityDevice") { return true }
        return false
        #else
        return false
        #endif
    }

    /// Check if the device has an M-series chip (Apple Silicon)
    static var hasMSeriesChip: Bool {
        #if os(macOS)
        var sysinfo = utsname()
        uname(&sysinfo)
        let machine = withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        }
        return machine?.contains("arm64") ?? false
        #elseif targetEnvironment(macCatalyst)
        return true
        #else
        // Vision Pro always has M-series chip
        if isRunningOnVisionPro { return true }

        let identifier = machineIdentifier

        // RealityDevice = Vision Pro (M2 or M5)
        if identifier.hasPrefix("RealityDevice") { return true }

        // Use chipNameFromIdentifier to check — avoids hardcoded list going stale
        let chip = chipNameFromIdentifier(identifier).lowercased()
        return chip.contains(" m1") || chip.contains(" m2") || chip.contains(" m3")
            || chip.contains(" m4") || chip.contains(" m5") || chip.contains("m-series")
        #endif
    }

    /// Check if device requires weight file size limit (1GB per weight file)
    static var requiresWeightSizeLimit: Bool {
        switch current {
        case .iPhone:
            return true
        case .iPad:
            return !hasMSeriesChip
        case .mac, .macCatalyst, .visionPro:
            return false
        case .other:
            return true
        }
    }

    /// Maximum weight file size in bytes (1GB for limited devices)
    static let maxWeightFileSize: Int64 = 1_073_741_824  // 1 GB

    // [ANE-COMPAT:M1-A14] Global attention compatibility check
    /// Whether this device supports Gemma3 global attention (KV-cache rotation).
    /// M1 Macs and A14 (and earlier) iOS devices cannot handle the global attention
    /// KV-cache implementation. Both M1 and A14 are first-gen Apple Silicon families
    /// and share this limitation.
    static var supportsGlobalAttention: Bool {
        let chip = chipName.lowercased()
        // M1 (including M1 Pro, M1 Max, M1 Ultra) does NOT support it
        if chip.range(of: #"\bm1\b"#, options: .regularExpression) != nil { return false }
        if chip.range(of: #"\bm1 "#, options: .regularExpression) != nil { return false }
        // A14 and earlier don't support it
        if chip.range(of: #"\ba1[0-4]\b"#, options: .regularExpression) != nil { return false }
        if chip.range(of: #"\ba[1-9]\b"#, options: .regularExpression) != nil { return false }
        // Intel Macs
        if chip.contains("intel") { return false }
        // Unknown chips — be conservative
        if chip.contains("unknown") { return false }
        // Everything else (M2+, A15+) supports it
        return true
    }

    // MARK: - Device Info

    /// Machine identifier (e.g., "arm64", "iPad14,5", "iPhone16,1")
    static var machineIdentifier: String {
        var sysinfo = utsname()
        uname(&sysinfo)
        return withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        } ?? "unknown"
    }

    /// CPU/chip brand string (e.g., "Apple M1", "Apple M4 Pro")
    /// On macOS uses sysctl; on iOS/visionOS infers from device identifier
    static var chipName: String {
        #if os(macOS)
        var size: size_t = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        if size > 0 {
            var buffer = [CChar](repeating: 0, count: size)
            sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
            let brand = String(cString: buffer).trimmingCharacters(in: .whitespacesAndNewlines)
            if !brand.isEmpty { return brand }
        }
        return hasMSeriesChip ? "Apple Silicon" : "Intel"
        #else
        let machine = machineIdentifier
        var result = chipNameFromIdentifier(machine)

        if isRunningOnVisionPro {
            // Vision Pro has at least M2 — if we got "arm64" / "Apple Silicon" or
            // something below M2 (shouldn't happen, but be safe), floor to M2
            let lower = result.lowercased()
            if lower.contains("apple silicon") || lower.contains("unknown")
                || lower.contains("a-series") || lower.contains(" m1") {
                result = "Apple M2 (Vision Pro)"
            }
        }
        return result
        #endif
    }

    /// Physical memory in bytes
    static var physicalMemory: UInt64 {
        ProcessInfo.processInfo.physicalMemory
    }

    /// Physical memory formatted (e.g., "8 GB")
    static var physicalMemoryString: String {
        ByteCountFormatter.string(fromByteCount: Int64(physicalMemory), countStyle: .memory)
    }

    /// Number of active processor cores
    static var processorCount: Int {
        ProcessInfo.processInfo.activeProcessorCount
    }

    /// OS version string
    static var osVersionString: String {
        let os = ProcessInfo.processInfo.operatingSystemVersion
        #if os(macOS)
        return "macOS \(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"
        #elseif os(visionOS)
        return "visionOS \(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"
        #else
        // iPad compat mode on Vision Pro still reports iPadOS — annotate it
        let device = UIDevice.current
        let base = "\(device.systemName) \(device.systemVersion)"
        if isRunningOnVisionPro {
            return "\(base) (on visionOS)"
        }
        return base
        #endif
    }

    /// Full device summary for logging
    static var deviceSummary: String {
        let chip = chipName
        let mem = physicalMemoryString
        let cores = processorCount
        let os = osVersionString
        let machine = machineIdentifier
        let mSeries = hasMSeriesChip ? "M-series" : "non-M-series"
        return "\(os) | \(chip) (\(mSeries)) | \(cores) cores | \(mem) RAM | \(machine)"
    }

    /// Parse "iPhone17,3" → (major: 17, minor: 3)
    private static func parseDeviceIdentifier(_ id: String, prefix: String) -> (major: Int, minor: Int)? {
        let body = id.replacingOccurrences(of: prefix, with: "")
        let parts = body.split(separator: ",")
        guard let major = parts.first.flatMap({ Int($0) }) else { return nil }
        let minor = parts.count > 1 ? (Int(parts[1]) ?? 1) : 1
        return (major, minor)
    }

    /// Infer Apple chip name from iOS/iPadOS device identifier.
    ///
    /// Mapping sources: appledb.dev, support.apple.com/en-us/108044, support.apple.com/en-us/108043
    /// iPhone major,minor → SoC:
    ///   18,1-2 = A19 Pro  |  18,3-4 = A19
    ///   17,1-2 = A18 Pro  |  17,3-5 = A18
    ///   16,1-2 = A17 Pro  |  15,x   = A16
    ///   14,x   = A15      |  13,x   = A14
    ///   12,x   = A13      |  11,x   = A12
    ///   10,x   = A11
    /// iPad major,minor → SoC:
    ///   16,3-6 = M4       |  16,1-2 = A17 Pro (iPad mini 7)
    ///   15,3-6 = M3       |  15,7-8 = A16 (iPad 11th gen)
    ///   14,3-6 = M2       |  14,8-11 = M2 (Air)  |  14,1-2 = A15 (mini 6)
    ///   13,4-11 = M1      |  13,16-19 = M1/A14 (Air 5/iPad 10)  |  13,1-2 = A14 (Air 4)
    ///   12,x   = A14      |  11,x = A12/A12X  |  8,x = A12X/A12Z
    private static func chipNameFromIdentifier(_ id: String) -> String {
        // MARK: iPhone
        if id.hasPrefix("iPhone"),
           let parsed = parseDeviceIdentifier(id, prefix: "iPhone") {
            switch parsed.major {
            case 18:
                return parsed.minor <= 2 ? "Apple A19 Pro" : "Apple A19"
            case 17:
                return parsed.minor <= 2 ? "Apple A18 Pro" : "Apple A18"
            case 16:
                return "Apple A17 Pro"
            case 15:
                return "Apple A16"
            case 14:
                return "Apple A15"
            case 13:
                return "Apple A14"
            case 12:
                return "Apple A13"
            case 11:
                return "Apple A12"
            case 10:
                return "Apple A11"
            default:
                // Future-proof: major > 18 → assume newer A-series
                return parsed.major > 18 ? "Apple A-series (new)" : "Apple A-series"
            }
        }

        // MARK: iPad
        if id.hasPrefix("iPad"),
           let parsed = parseDeviceIdentifier(id, prefix: "iPad") {
            switch parsed.major {
            case 16:
                // 16,1-2 = iPad mini 7 (A17 Pro), 16,3-6 = iPad Pro M4
                return parsed.minor <= 2 ? "Apple A17 Pro" : "Apple M4"
            case 15:
                // 15,3-6 = iPad Air M3, 15,7-8 = iPad 11th gen (A16)
                return parsed.minor <= 6 ? "Apple M3" : "Apple A16"
            case 14:
                // 14,1-2 = iPad mini 6 (A15), 14,3-6 = iPad Pro M2, 14,8-11 = iPad Air M2
                if parsed.minor <= 2 { return "Apple A15" }
                return "Apple M2"
            case 13:
                // 13,1-2 = iPad Air 4 (A14), 13,4-11 = iPad Pro M1
                // 13,16-17 = iPad Air 5 (M1), 13,18-19 = iPad 10th gen (A14)
                if parsed.minor <= 2 { return "Apple A14" }
                if parsed.minor <= 11 { return "Apple M1" }
                if parsed.minor <= 17 { return "Apple M1" }
                return "Apple A14"
            case 12:
                return "Apple A14"
            case 11:
                return "Apple A12/A12X"
            case 8:
                return "Apple A12X/A12Z"
            case 7:
                return "Apple A10"
            default:
                return parsed.major > 16 ? "Apple M-series (new)" : "Apple A-series"
            }
        }

        // MARK: Apple Vision Pro
        // RealityDevice14,1 = Vision Pro (M2), RealityDevice15,x = Vision Pro M5
        if id.hasPrefix("RealityDevice"),
           let parsed = parseDeviceIdentifier(id, prefix: "RealityDevice") {
            switch parsed.major {
            case 14:
                return "Apple M2"
            case 15:
                return "Apple M5"
            default:
                return parsed.major > 15 ? "Apple M-series (new)" : "Apple M2"
            }
        }

        if id.contains("arm64") { return "Apple Silicon" }
        return "Unknown (\(id))"
    }
}

enum LocalModelImportMode: String, CaseIterable, Sendable {
    case importCopy
    case linkExternal

    var displayTitle: String {
        switch self {
        case .importCopy: return "Import (Copy)"
        case .linkExternal: return "Link (External)"
        }
    }
}

struct LocalModelInspection: Sendable {
    let droppedURL: URL
    let modelRootURL: URL
    let suggestedDisplayName: String
    let suggestedModelId: String
}

enum LocalModelValidationError: LocalizedError {
    case notDirectory
    case invalidStructure([String])

    var errorDescription: String? {
        switch self {
        case .notDirectory:
            return "Please drop a folder, not a file."
        case .invalidStructure(let issues):
            if issues.isEmpty {
                return "Folder does not look like a valid model root."
            }
            return issues.joined(separator: " ")
        }
    }
}

enum ModelPackageImportError: LocalizedError {
    case invalidPackageRoot
    case missingManifest
    case invalidManifest
    case unsupportedFormatVersion(Int)
    case incompatibleAppVersion(minimum: String, current: String)
    case missingFile(path: String)
    case fileHashMismatch(path: String)
    case invalidModelRoot

    var errorDescription: String? {
        switch self {
        case .invalidPackageRoot:
            return "Could not resolve model package folder."
        case .missingManifest:
            return "Package is missing manifest.json."
        case .invalidManifest:
            return "Package manifest is invalid."
        case .unsupportedFormatVersion(let version):
            return "Unsupported package format version: \(version)."
        case .incompatibleAppVersion(let minimum, let current):
            return "Package requires app version \(minimum)+. Current version is \(current)."
        case .missingFile(let path):
            return "Package file is missing: \(path)."
        case .fileHashMismatch(let path):
            return "Hash validation failed for \(path)."
        case .invalidModelRoot:
            return "Package does not contain a valid model root."
        }
    }
}

/// View model for managing models
@Observable
@MainActor
final class ModelManagerViewModel {
    // MARK: - State

    /// All available models
    var availableModels: [ModelInfo] = []

    /// Currently downloading model ID
    var downloadingModelId: String?

    /// Current download progress
    var downloadProgress: DownloadProgress?

    /// Currently loaded model ID
    var loadedModelId: String?

    /// Recently added model ID (used by UI to highlight new entries)
    var justAddedModelId: String?

    /// Model loading progress
    var loadingProgress: ModelLoadingProgress?

    /// Error message
    var errorMessage: String?

    /// Non-error status shown while/after importing an incoming package.
    var incomingTransferStatusMessage: String?

    /// Indicates that a package import is currently running.
    var isImportingIncomingPackage: Bool = false

    /// Signals UI to focus/open model list after a package import.
    var lastImportedPackageModelId: String?

    private var recentlyHandledIncomingURLKeys: [String: Date] = [:]

    /// Signal from child views (e.g. ChatView) to open the model selection sheet.
    var requestModelSelection: Bool = false

    /// Whether the initial model list load + auto-load attempt has completed.
    /// Used to suppress the "Download or Select Model" prompt until startup finishes.
    var hasCompletedInitialLoad: Bool = false

    /// Whether model is being loaded
    var isLoadingModel: Bool = false

    /// ID of model currently being loaded
    var loadingModelId: String?

    #if os(macOS)
    /// Whether a model is being packaged for sharing
    var isSharingModel: Bool = false

    /// ID of model currently being shared
    var sharingModelId: String?
    #endif

    private var clearJustAddedTask: Task<Void, Never>?
    private var modelLoadTask: Task<Void, Never>?
    #if os(macOS)
    private var preparedTransferPackages: [URL] = []
    #endif

    // MARK: - Computed Properties

    /// Downloaded models
    var downloadedModels: [ModelInfo] {
        availableModels.filter { $0.isDownloaded }
    }

    /// Models available for download (excludes errored models)
    var availableForDownload: [ModelInfo] {
        availableModels.filter {
            $0.sourceKind == .huggingFace &&
            !$0.isDownloaded &&
            !$0.isDownloading &&
            $0.downloadError == nil
        }
    }

    /// Total size of downloaded models
    var downloadedModelsSize: String {
        let total = downloadedModels.compactMap { $0.sizeBytes }.reduce(0, +)
        return ByteCountFormatter.string(fromByteCount: total, countStyle: .file)
    }

    /// Display name of model currently being loaded
    var loadingModelName: String? {
        guard let loadingId = loadingModelId else { return nil }
        return availableModels.first { $0.id == loadingId }?.name
    }

    // MARK: - Initialization

    init() {
        // Immediately show default models (before async check completes)
        availableModels = ModelInfo.defaultModels
        logInfo("ModelManagerViewModel init: set \(availableModels.count) default models", category: .model)

        // Then async check download status and load custom models
        Task {
            await loadModels()
        }
    }

    // MARK: - Model Loading

    /// HuggingFace collection URL for dynamic model list
    private static let collectionURL = URL(string: "https://huggingface.co/api/collections/anemll/anemll-chat")!

    /// Load model list: try HF collection → cache → hardcoded defaults, then merge custom models
    func loadModels() async {
        logInfo("Loading models...", category: .model)

        // Step 1: Get base model list (collection → cache → defaults)
        var models: [ModelInfo]

        do {
            let fetched = try await fetchCollectionModels()
            models = fetched
            logInfo("Fetched \(fetched.count) models from HuggingFace collection", category: .model)
            // Cache for offline use
            await StorageService.shared.saveCollectionCache(fetched)
        } catch {
            print("[Collection] ERROR: \(error)")
            logWarning("Failed to fetch collection: \(error)", category: .model)
            // Try cache
            if let cached = await StorageService.shared.loadCollectionCache() {
                models = cached
                logInfo("Using \(cached.count) cached collection models", category: .model)
            } else {
                models = ModelInfo.defaultModels
                logInfo("Using \(models.count) hardcoded default models", category: .model)
            }
        }

        // Step 2: Add custom models from registry (local imports/links and downloaded HF models only)
        // The collection fetch (Step 1) is the authoritative source for HuggingFace models.
        // Only merge registry entries that are locally imported/linked or actually downloaded.
        // This prevents stale HF entries (from old author-search) from polluting the model list.
        do {
            let registryModels = try await StorageService.shared.loadModelsRegistry()
            for entry in registryModels {
                guard !models.contains(where: { $0.id == entry.id }) else { continue }
                // Keep: local imports, local links, and downloaded HF models
                // Skip: non-downloaded HF models (the collection already covers those)
                if entry.sourceKind == .localImported || entry.sourceKind == .localLinked || entry.isDownloaded {
                    models.append(entry)
                    logDebug("Added custom model: \(entry.id)", category: .model)
                } else {
                    logDebug("Skipped stale registry entry: \(entry.id)", category: .model)
                }
            }
        } catch {
            logWarning("Failed to load custom models: \(error)", category: .model)
        }

        // Step 3: Check model availability and reset stale download state.
        // Show models immediately so the UI isn't blank while linked models are checked.
        availableModels = models

        // Check non-linked models first (fast, local-only I/O)
        for i in models.indices where models[i].sourceKind != .localLinked {
            models[i] = await refreshedModelStatus(for: models[i])
        }
        availableModels = models

        // Check linked models (may involve bookmark resolution + network I/O, each with a timeout)
        for i in models.indices where models[i].sourceKind == .localLinked {
            models[i] = await refreshedModelStatus(for: models[i])
            // Update UI progressively so each result appears as soon as it's ready
            availableModels = models
        }

        refreshGlobalAttentionCompatibilityCache()
        logInfo("Loaded \(models.count) models (\(downloadedModels.count) downloaded)", category: .model)

        // Auto-load last model after models are loaded (if setting enabled)
        if await StorageService.shared.autoLoadLastModel {
            await autoLoadLastModel()
        }

        hasCompletedInitialLoad = true
    }

    // MARK: - HuggingFace Collection Fetch

    /// Fetch model list from HuggingFace collection API
    private func fetchCollectionModels() async throws -> [ModelInfo] {
        let collectionURL = Self.collectionURL
        print("[Collection] Fetching from: \(collectionURL)")

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 10
        let session = URLSession(configuration: config)

        let (collectionData, response) = try await session.data(from: collectionURL)
        if let httpResponse = response as? HTTPURLResponse {
            print("[Collection] HTTP \(httpResponse.statusCode), \(collectionData.count) bytes")
        }

        // Decode collection response
        struct CollectionItem: Decodable {
            let id: String
            let type: String?
            let position: Int?
        }
        struct CollectionResponse: Decodable {
            let items: [CollectionItem]
        }

        let collection = try JSONDecoder().decode(CollectionResponse.self, from: collectionData)
        let modelItems = collection.items
            .filter { ($0.type ?? "model") == "model" }
            .sorted { ($0.position ?? 0) < ($1.position ?? 0) }

        print("[Collection] Found \(modelItems.count) models in collection")

        // Fetch per-model details concurrently
        let models = await withTaskGroup(of: (Int, ModelInfo?).self) { group in
            for (index, item) in modelItems.enumerated() {
                group.addTask {
                    do {
                        let modelInfo = try await Self.fetchModelDetail(id: item.id, session: session)
                        print("[Collection] Fetched detail for: \(item.id)")
                        return (index, modelInfo)
                    } catch {
                        print("[Collection] Failed detail for \(item.id): \(error)")
                        // Fall back to parsing from repo ID only
                        return (index, ModelInfo.fromHuggingFaceRepo(id: item.id, sizeBytes: nil, modelType: nil))
                    }
                }
            }

            var results: [(Int, ModelInfo)] = []
            for await result in group {
                if let model = result.1 {
                    results.append((result.0, model))
                }
            }
            return results.sorted { $0.0 < $1.0 }.map(\.1)
        }

        print("[Collection] Returning \(models.count) models")
        return models
    }

    /// Fetch individual model details from HuggingFace API
    private static func fetchModelDetail(id: String, session: URLSession) async throws -> ModelInfo {
        struct SiblingFile: Decodable {
            let size: Int64?
        }
        struct ModelDetailConfig: Decodable {
            let model_type: String?
        }
        struct ModelDetailResponse: Decodable {
            let siblings: [SiblingFile]?
            let config: ModelDetailConfig?
        }

        // Use blobs=true to get actual file sizes from siblings
        let url = URL(string: "https://huggingface.co/api/models/\(id)?blobs=true")!
        let (data, _) = try await session.data(from: url)
        let detail = try JSONDecoder().decode(ModelDetailResponse.self, from: data)

        // Sum actual file sizes instead of using usedStorage (which includes git LFS history)
        let totalSize: Int64? = detail.siblings?.compactMap(\.size).reduce(0, +)

        return ModelInfo.fromHuggingFaceRepo(
            id: id,
            sizeBytes: totalSize,
            modelType: detail.config?.model_type
        )
    }

    /// Refresh download status for all models and discover new ones from HuggingFace collection
    func refreshModelStatus() async {
        // Discover new models from HuggingFace curated collection (not author search)
        var discovered: [ModelInfo] = []
        do {
            discovered = try await fetchCollectionModels()
        } catch {
            logWarning("Failed to fetch collection during refresh: \(error)", category: .model)
        }
        var models = availableModels

        // Merge newly discovered models
        var addedCount = 0
        for newModel in discovered {
            if !models.contains(where: { $0.id == newModel.id }) {
                models.append(newModel)
                addedCount += 1
                logInfo("Discovered new model: \(newModel.id)", category: .model)
            }
        }
        if addedCount > 0 {
            logInfo("Discovered \(addedCount) new model(s) from HuggingFace", category: .model)
        }

        // Refresh download status for all models (non-linked first, then linked with timeout)
        var refreshedModels: [ModelInfo] = []
        refreshedModels.reserveCapacity(models.count)

        for model in models {
            refreshedModels.append(await refreshedModelStatus(for: model))
            // Update UI progressively for linked models (may have timeout delays)
            if model.sourceKind == .localLinked {
                availableModels = refreshedModels + Array(models.dropFirst(refreshedModels.count))
            }
        }
        availableModels = refreshedModels
        refreshGlobalAttentionCompatibilityCache()

        // Save registry so newly discovered models persist
        if addedCount > 0 {
            try? await StorageService.shared.saveModelsRegistry(availableModels)
        }

        lastRefreshDiscoveredCount = addedCount
    }

    /// Number of models discovered in last refresh (for UI feedback)
    var lastRefreshDiscoveredCount: Int = 0

    // MARK: - Download

    // [ANE-COMPAT:M1-A14] Pre-download compatibility warning state
    /// Warning message shown before downloading an incompatible model
    var downloadCompatibilityWarningMessage: String?
    /// Whether to show the pre-download compatibility warning alert
    var showDownloadCompatibilityWarningAlert: Bool = false
    /// Model pending download (waiting for user confirmation after compatibility check)
    private var pendingDownloadModel: ModelInfo?

    /// Download a model (public API - checks compatibility then starts download)
    func downloadModel(_ model: ModelInfo) async {
        guard model.sourceKind == .huggingFace else {
            errorMessage = "Only HuggingFace models support download."
            return
        }
        guard !model.isDownloaded, !model.isDownloading else { return }

        // [ANE-COMPAT:M1-A14] On M1/A14, fetch meta.yaml first to check compatibility
        if !DeviceType.supportsGlobalAttention {
            if let warning = await fetchAndCheckPreDownloadCompatibility(for: model) {
                downloadCompatibilityWarningMessage = warning
                pendingDownloadModel = model
                showDownloadCompatibilityWarningAlert = true
                logWarning("Pre-download compatibility warning for \(model.id): \(warning)", category: .download)
                return
            }
        }

        // No compatibility issue — start download immediately
        await startDownload(model)
    }

    /// Continue downloading after user confirmed pre-download compatibility warning
    func confirmDownloadModel() async {
        guard let model = pendingDownloadModel else {
            clearPendingDownloadState()
            return
        }
        clearPendingDownloadState()
        await startDownload(model)
    }

    /// Cancel download after pre-download compatibility warning
    func cancelDownloadModel() {
        clearPendingDownloadState()
    }

    private func clearPendingDownloadState() {
        pendingDownloadModel = nil
        showDownloadCompatibilityWarningAlert = false
        downloadCompatibilityWarningMessage = nil
    }

    /// Internal: set UI state and kick off performDownload
    private func startDownload(_ model: ModelInfo) async {
        downloadingModelId = model.id
        updateModelDownloading(model.id, isDownloading: true)
        await performDownload(model)
    }

    // [ANE-COMPAT:M1-A14] Fetch meta.yaml from HuggingFace and check for global attention incompatibility
    /// Returns a warning message if the model is incompatible, or nil if OK / unknown.
    /// Caches the result so ModelCard can display "Not Compatible" even before download.
    private func fetchAndCheckPreDownloadCompatibility(for model: ModelInfo) async -> String? {
        logInfo("Fetching meta.yaml for pre-download compatibility check: \(model.id)", category: .download)

        guard let yamlContent = await DownloadService.shared.fetchMetaYaml(for: model.id) else {
            logDebug("No meta.yaml available for pre-download check — proceeding with download", category: .download)
            return nil
        }

        logInfo("Pre-download meta.yaml fetched for \(model.id), checking compatibility...", category: .download)

        // Parse the YAML content using ModelMetadata's parsing logic
        guard let metadata = ModelMetadata.loadFromString(yamlContent) else {
            logDebug("Failed to parse meta.yaml for pre-download check — proceeding with download", category: .download)
            return nil
        }

        // Only Gemma architecture is affected
        let isGemma = metadata.isGemmaFamily
            || model.architecture?.lowercased() == "gemma"
            || model.id.lowercased().contains("gemma")
        guard isGemma else {
            // Cache as "checked, no warning" so we don't re-fetch
            cacheGlobalAttentionWarning(nil, for: model.id)
            return nil
        }

        // When contextLength > slidingWindow, global attention layers are needed
        guard let sw = metadata.slidingWindow, metadata.contextLength > sw else {
            cacheGlobalAttentionWarning(nil, for: model.id)
            return nil
        }

        let warning = "This Gemma model uses global attention (context \(metadata.contextLength) > SWA \(sw)) which is not compatible with \(DeviceType.chipName). Requires Apple M2+ (Mac) or A15+ (iPhone/iPad). You can still download it, but it may not load correctly."
        // Cache so ModelCard shows "Not Compatible" tag immediately
        cacheGlobalAttentionWarning(warning, for: model.id)
        return warning
    }

    /// Cancel ongoing download
    func cancelDownload() async {
        guard let modelId = downloadingModelId else { return }

        await DownloadService.shared.cancelDownload(modelId)
        updateModelDownloading(modelId, isDownloading: false)

        downloadingModelId = nil
        downloadProgress = nil

        logInfo("Download cancelled: \(modelId)", category: .download)
    }

    /// Delete a downloaded model
    func deleteModel(_ model: ModelInfo) async {
        do {
            if model.sourceKind != .localLinked {
                try await StorageService.shared.deleteModel(model.id)
            }

            if let index = availableModels.firstIndex(where: { $0.id == model.id }) {
                // Local-only models should be removed from list on delete.
                if model.sourceKind == .localImported || model.sourceKind == .localLinked {
                    availableModels.remove(at: index)
                } else {
                    availableModels[index].isDownloaded = false
                    availableModels[index].localPath = nil
                    availableModels[index].metaYamlPath = nil
                }
            }
            clearCachedGlobalAttentionWarning(for: model.id)

            try? await StorageService.shared.saveModelsRegistry(availableModels)

            // Unload if currently loaded
            if loadedModelId == model.id {
                loadedModelId = nil
            }

            logInfo("Deleted model: \(model.id)", category: .model)

        } catch {
            errorMessage = error.localizedDescription
            logError("Failed to delete model: \(error)", category: .model)
        }
    }

    // MARK: - Model Loading

    /// Warning message shown before loading a model with oversized weights
    var weightWarningMessage: String?

    /// Whether to show the weight warning alert
    var showWeightWarningAlert: Bool = false

    // [ANE-COMPAT:M1-A14] Compatibility warning state
    /// Warning message shown before loading an incompatible model
    var compatibilityWarningMessage: String?

    /// Whether to show the compatibility warning alert
    var showCompatibilityWarningAlert: Bool = false

    /// Model pending load (waiting for user confirmation)
    private var pendingLoadModel: ModelInfo?

    /// Which warning gate the user is currently confirming before load.
    private enum PendingLoadWarning {
        case none
        case weightSize
        case compatibility
    }
    private var pendingLoadWarning: PendingLoadWarning = .none

    // [ANE-COMPAT:M1-A14] Cache post-download compatibility checks
    private var globalAttentionWarningByModelId: [String: String] = [:]
    private var globalAttentionCheckedModelIds: Set<String> = []

    /// Load a model for inference
    func loadModelForInference(_ model: ModelInfo) async {
        // For linked models, resolve bookmark to get security-scoped access
        let path: String
        if model.sourceKind == .localLinked {
            guard let url = resolveLinkedModelURL(for: model) else {
                errorMessage = "Linked source folder is unavailable. Re-link or re-import this model."
                return
            }
            path = url.path
            // Note: security-scoped access stays active until stopAccessingSecurityScopedResource()
            // is called — performModelLoad will use the path while access is held.
        } else {
            guard model.isDownloaded, let localPath = model.localPath else {
                errorMessage = "Model not downloaded"
                return
            }
            path = localPath
        }

        // Reset stale warning state from a previous load attempt.
        clearPendingLoadState()

        // Check for weight size warning
        if let warning = getWeightSizeWarning(for: model) {
            weightWarningMessage = warning
            pendingLoadModel = model
            pendingLoadWarning = .weightSize
            showWeightWarningAlert = true
            logWarning("Model has weight size warning: \(warning)", category: .model)
            return
        }

        // [ANE-COMPAT:M1-A14] Check for global attention compatibility
        if let compatWarning = getGlobalAttentionWarning(for: model) {
            compatibilityWarningMessage = compatWarning
            pendingLoadModel = model
            pendingLoadWarning = .compatibility
            showCompatibilityWarningAlert = true
            logWarning("Model has compatibility warning: \(compatWarning)", category: .model)
            return
        }

        await performModelLoad(model, path: path)
    }

    /// Continue loading model after user confirmed weight warning
    func confirmLoadModel() async {
        guard let model = pendingLoadModel else {
            clearPendingLoadState()
            return
        }

        // Resolve path for linked models via bookmark
        let path: String
        if model.sourceKind == .localLinked {
            guard let url = resolveLinkedModelURL(for: model) else {
                errorMessage = "Linked source folder is unavailable. Re-link or re-import this model."
                clearPendingLoadState()
                return
            }
            path = url.path
        } else {
            guard let localPath = model.localPath else {
                clearPendingLoadState()
                return
            }
            path = localPath
        }

        switch pendingLoadWarning {
        case .weightSize:
            showWeightWarningAlert = false
            weightWarningMessage = nil

            // If weights are acknowledged, still enforce Gemma global-attention warning.
            if let compatWarning = getGlobalAttentionWarning(for: model) {
                compatibilityWarningMessage = compatWarning
                showCompatibilityWarningAlert = true
                pendingLoadWarning = .compatibility
                logWarning("Model has compatibility warning: \(compatWarning)", category: .model)
                return
            }

            pendingLoadModel = nil
            pendingLoadWarning = .none
            await performModelLoad(model, path: path)

        case .compatibility, .none:
            clearPendingLoadState()
            await performModelLoad(model, path: path)
        }
    }

    /// Cancel model load
    func cancelLoadModel() {
        clearPendingLoadState()
    }

    private func clearPendingLoadState() {
        pendingLoadModel = nil
        pendingLoadWarning = .none
        showWeightWarningAlert = false
        showCompatibilityWarningAlert = false
        weightWarningMessage = nil
        compatibilityWarningMessage = nil
    }

    /// Cancel an in-progress model load
    func cancelModelLoading() async {
        modelLoadTask?.cancel()
        modelLoadTask = nil
        await InferenceService.shared.unloadModel()
        isLoadingModel = false
        loadingModelId = nil
        loadingProgress = nil
        logInfo("Model loading cancelled by user", category: .model)
    }

    /// Internal method to perform model loading
    private func performModelLoad(_ model: ModelInfo, path: String) async {
        isLoadingModel = true
        loadingModelId = model.id
        loadingProgress = nil
        errorMessage = nil

        // Start a task to poll InferenceService's loading progress
        let progressTask = Task { @MainActor in
            while !Task.isCancelled && isLoadingModel {
                loadingProgress = InferenceService.shared.loadingProgress
                try? await Task.sleep(for: .milliseconds(100))
            }
        }

        modelLoadTask = Task { @MainActor in
            do {
                let modelURL = URL(fileURLWithPath: path)
                try await InferenceService.shared.loadModel(from: modelURL)

                guard !Task.isCancelled else { return }

                loadedModelId = model.id
                await StorageService.shared.saveSelectedModelId(model.id)
                logInfo("Model loaded: \(model.id)", category: .model)

            } catch {
                guard !Task.isCancelled else { return }
                loadedModelId = nil
                errorMessage = error.localizedDescription
                logError("Failed to load model: \(error)", category: .model)
            }

            progressTask.cancel()
            isLoadingModel = false
            loadingModelId = nil
            loadingProgress = nil
        }

        await modelLoadTask?.value
        modelLoadTask = nil
    }

    /// Unload the current model
    func unloadCurrentModel() async {
        await InferenceService.shared.unloadModel()
        loadedModelId = nil
        logInfo("Model unloaded", category: .model)
    }

    /// Auto-load the last selected model
    func autoLoadLastModel() async {
        guard let selectedId = await StorageService.shared.selectedModelId else {
            logInfo("[AUTO-LOAD] No saved model ID found", category: .model)
            return
        }

        logInfo("[AUTO-LOAD] Looking for model: \(selectedId)", category: .model)

        if let model = availableModels.first(where: { $0.id == selectedId && $0.isDownloaded }) {
            logInfo("[AUTO-LOAD] Found model, loading: \(model.name)", category: .model)
            await loadModelForInference(model)
        } else {
            logWarning("[AUTO-LOAD] Model not found or not downloaded: \(selectedId)", category: .model)
        }
    }

    // MARK: - Custom Models

    /// Add a custom model from URL and start download
    /// NOTE: This returns immediately after adding the model - download runs in background
    func addCustomModel(repoId: String, name: String) async {
        // Trim whitespace from inputs to prevent path issues
        let cleanRepoId = repoId.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanName = name.trimmingCharacters(in: .whitespacesAndNewlines)

        logInfo("addCustomModel called: '\(cleanRepoId)'", category: .model)

        // Validate repo ID format
        guard !cleanRepoId.isEmpty, cleanRepoId.contains("/") else {
            errorMessage = "Invalid repository ID format. Use: owner/repo-name"
            logError("Invalid repo ID format: '\(cleanRepoId)'", category: .model)
            return
        }

        // Check if model already exists
        if let existingIndex = availableModels.firstIndex(where: { $0.id == cleanRepoId }) {
            let existingModel = availableModels[existingIndex]

            // If already downloaded, just inform user
            if existingModel.isDownloaded {
                errorMessage = "Model already downloaded: \(cleanName)"
                logInfo("Model already downloaded: \(cleanRepoId)", category: .model)
                return
            }

            // If currently downloading, don't start another
            if existingModel.isDownloading {
                logInfo("Model already downloading: \(cleanRepoId)", category: .model)
                return
            }

            // Model exists but not downloaded - start download
            logInfo("Starting download for existing model: \(cleanRepoId)", category: .model)
            markModelAsJustAdded(existingModel.id)

            // Mark as downloading IMMEDIATELY so UI shows it
            downloadingModelId = existingModel.id
            updateModelDownloading(existingModel.id, isDownloading: true)

            Task {
                await performDownload(existingModel)
            }
            return
        }

        // New model - add to list
        let model = ModelInfo(
            id: cleanRepoId,
            name: cleanName.isEmpty ? cleanRepoId.components(separatedBy: "/").last ?? cleanRepoId : cleanName,
            description: "Custom model from HuggingFace",
            size: "Unknown"
        )

        availableModels.append(model)
        markModelAsJustAdded(model.id)
        logInfo("Added model to list: \(cleanRepoId), total models: \(availableModels.count)", category: .model)

        // Save to registry FIRST (before download starts)
        do {
            try await StorageService.shared.saveModelsRegistry(availableModels)
            logInfo("Saved model registry with \(availableModels.count) models", category: .model)
        } catch {
            logError("Failed to save model registry: \(error)", category: .model)
        }

        // Mark as downloading IMMEDIATELY so UI shows it in Downloading section
        // This must happen BEFORE the async Task to avoid UI timing gap
        downloadingModelId = model.id
        updateModelDownloading(model.id, isDownloading: true)

        // Start download in background (don't await - return immediately)
        Task {
            await performDownload(model)
        }
    }

    // MARK: - Local Models (macOS import/link workflow)

    /// Inspect a dropped/selected folder and derive a deterministic local model proposal.
    func inspectLocalModelFolder(_ droppedURL: URL) throws -> LocalModelInspection {
        let modelRootURL = try detectLocalModelRoot(from: droppedURL)
        let baseName = suggestModelName(from: droppedURL)
        let uniqueName = uniqueDisplayName(for: baseName)
        let uniqueModelId = uniqueLocalModelId(forDisplayName: uniqueName)

        return LocalModelInspection(
            droppedURL: droppedURL,
            modelRootURL: modelRootURL,
            suggestedDisplayName: uniqueName,
            suggestedModelId: uniqueModelId
        )
    }

    /// Add a local model either by copying into app storage or linking the external folder.
    func addLocalModel(from droppedURL: URL, displayName: String, mode: LocalModelImportMode) async {
        errorMessage = nil

        // Start security-scoped access for the picker URL (needed for sandboxed apps).
        let accessGranted = droppedURL.startAccessingSecurityScopedResource()
        defer { if accessGranted { droppedURL.stopAccessingSecurityScopedResource() } }

        do {
            let inspection = try inspectLocalModelFolder(droppedURL)
            let requestedName = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
            let finalName = uniqueDisplayName(for: requestedName.isEmpty ? inspection.suggestedDisplayName : requestedName)
            let finalModelId = uniqueLocalModelId(forDisplayName: finalName)
            let rootPath = inspection.modelRootURL.path

            if availableModels.contains(where: { $0.localPath == rootPath || $0.linkedPath == rootPath }) {
                errorMessage = "This model folder is already added."
                return
            }

            let localPath: String
            let linkedPath: String?
            let bookmarkDataBase64: String?
            let description: String

            switch mode {
            case .importCopy:
                let importedPath = try await StorageService.shared.importModelDirectory(from: inspection.modelRootURL, toModelId: finalModelId)
                localPath = importedPath.path
                linkedPath = nil
                bookmarkDataBase64 = nil
                description = "Local model (imported)"

            case .linkExternal:
                localPath = inspection.modelRootURL.path
                linkedPath = inspection.modelRootURL.path
                bookmarkDataBase64 = makeBookmarkDataBase64(for: inspection.modelRootURL)
                description = "Local model (linked)"
            }

            // Calculate actual model size
            let sizeBytes = calculateDirectorySize(at: URL(fileURLWithPath: localPath))
            let sizeString = sizeBytes > 0 ? ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file) : "Local"

            let newModel = ModelInfo(
                id: finalModelId,
                name: finalName,
                description: description,
                size: sizeString,
                sizeBytes: sizeBytes > 0 ? sizeBytes : nil,
                isDownloaded: true,
                localPath: localPath,
                metaYamlPath: URL(fileURLWithPath: localPath).appendingPathComponent("meta.yaml").path,
                sourceKind: mode == .importCopy ? .localImported : .localLinked,
                linkedPath: linkedPath,
                bookmarkDataBase64: bookmarkDataBase64
            )

            availableModels.append(newModel)
            markModelAsJustAdded(newModel.id)
            try await StorageService.shared.saveModelsRegistry(availableModels)
            logInfo("Added local model: \(newModel.id) (\(mode.rawValue))", category: .model)

        } catch {
            errorMessage = error.localizedDescription
            logError("Failed to add local model: \(error)", category: .model)
        }
    }

    // MARK: - Package Import (iOS receiver)

    func handleIncomingTransferURL(_ url: URL) async {
        guard shouldProcessIncomingTransferURL(url) else {
            logDebug("Ignoring duplicate incoming transfer URL: \(url.path)", category: .model)
            return
        }

        isImportingIncomingPackage = true
        incomingTransferStatusMessage = "Importing model package..."

        do {
            let imported = try await importModelPackage(from: url)
            incomingTransferStatusMessage = "Imported model: \(imported.name)"
            lastImportedPackageModelId = imported.id
            logInfo("Imported transferred model package: \(imported.id)", category: .model)
        } catch {
            errorMessage = error.localizedDescription
            logError("Failed to import transferred model package: \(error)", category: .model)
        }
        isImportingIncomingPackage = false
    }

    private func importModelPackage(from incomingURL: URL) async throws -> ModelInfo {
        let accessStarted = incomingURL.startAccessingSecurityScopedResource()
        defer {
            if accessStarted {
                incomingURL.stopAccessingSecurityScopedResource()
            }
        }

        let packageRoot = try resolveIncomingPackageRoot(from: incomingURL)
        let manifestURL = packageRoot.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelPackageImportError.missingManifest
        }

        let manifestData = try Data(contentsOf: manifestURL)
        guard let manifest = try? JSONDecoder().decode(ModelPackageManifest.self, from: manifestData) else {
            throw ModelPackageImportError.invalidManifest
        }

        try validatePackageCompatibility(manifest)
        try await validatePackageFiles(manifest: manifest, packageRoot: packageRoot)

        let modelRootURL: URL
        if let rootPath = manifest.modelRootPath, !rootPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            guard let normalizedRoot = Self.normalizePackageRelativePath(rootPath) else {
                throw ModelPackageImportError.invalidManifest
            }
            modelRootURL = packageRoot.appendingPathComponent(normalizedRoot, isDirectory: true)
        } else {
            modelRootURL = packageRoot
        }

        let resolvedModelRoot: URL
        do {
            resolvedModelRoot = try detectLocalModelRoot(from: modelRootURL)
        } catch {
            throw ModelPackageImportError.invalidModelRoot
        }

        let manifestName = manifest.modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        let preferredName = manifestName.isEmpty ? suggestModelName(from: resolvedModelRoot) : manifestName
        let finalName = uniqueDisplayName(for: preferredName)
        let finalModelId = uniqueLocalModelId(forDisplayName: finalName)

        let importedPath = try await StorageService.shared.importModelDirectory(from: resolvedModelRoot, toModelId: finalModelId)

        // Calculate actual model size
        let sizeBytes = calculateDirectorySize(at: importedPath)
        let sizeString = sizeBytes > 0 ? ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file) : "Local"

        let newModel = ModelInfo(
            id: finalModelId,
            name: finalName,
            description: "Transferred model package",
            size: sizeString,
            sizeBytes: sizeBytes > 0 ? sizeBytes : nil,
            isDownloaded: true,
            localPath: importedPath.path,
            metaYamlPath: importedPath.appendingPathComponent("meta.yaml").path,
            sourceKind: .localImported
        )

        availableModels.removeAll { $0.id == newModel.id }
        availableModels.append(newModel)
        markModelAsJustAdded(newModel.id)
        try await StorageService.shared.saveModelsRegistry(availableModels)
        cleanupIncomingPackageArtifactsIfNeeded(incomingURL: incomingURL, resolvedPackageRoot: packageRoot)
        return newModel
    }

    private func resolveIncomingPackageRoot(from incomingURL: URL) throws -> URL {
        let fileManager = FileManager.default
        let standardized = incomingURL.standardizedFileURL

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: standardized.path, isDirectory: &isDirectory) else {
            throw ModelPackageImportError.invalidPackageRoot
        }

        if isDirectory.boolValue {
            let manifestAtRoot = standardized.appendingPathComponent("manifest.json")
            if fileManager.fileExists(atPath: manifestAtRoot.path) {
                return standardized
            }

            let children = (try? fileManager.contentsOfDirectory(at: standardized, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])) ?? []
            if let childWithManifest = children.first(where: {
                var childIsDir: ObjCBool = false
                guard fileManager.fileExists(atPath: $0.path, isDirectory: &childIsDir), childIsDir.boolValue else { return false }
                return fileManager.fileExists(atPath: $0.appendingPathComponent("manifest.json").path)
            }) {
                return childWithManifest
            }
        }

        throw ModelPackageImportError.invalidPackageRoot
    }

    private func validatePackageCompatibility(_ manifest: ModelPackageManifest) throws {
        guard manifest.formatVersion == 1 else {
            throw ModelPackageImportError.unsupportedFormatVersion(manifest.formatVersion)
        }

        if let minAppVersion = manifest.minAppVersion, !minAppVersion.isEmpty {
            let currentVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
            if compareVersionStrings(currentVersion, minAppVersion) == .orderedAscending {
                throw ModelPackageImportError.incompatibleAppVersion(minimum: minAppVersion, current: currentVersion)
            }
        }
    }

    private func validatePackageFiles(manifest: ModelPackageManifest, packageRoot: URL) async throws {
        try await Task.detached(priority: .userInitiated) {
            let fileManager = FileManager.default
            for entry in manifest.files {
                guard let normalizedEntryPath = Self.normalizePackageRelativePath(entry.path) else {
                    throw ModelPackageImportError.invalidManifest
                }
                guard let fileURL = Self.resolvePackageFileURL(
                    for: normalizedEntryPath,
                    packageRoot: packageRoot,
                    fileManager: fileManager
                ) else {
                    throw ModelPackageImportError.missingFile(path: entry.path)
                }

                if let expectedSize = entry.sizeBytes,
                   let attrs = try? fileManager.attributesOfItem(atPath: fileURL.path),
                   let size = attrs[.size] as? Int64,
                   size != expectedSize {
                    throw ModelPackageImportError.fileHashMismatch(path: entry.path)
                }

                let data = try Data(contentsOf: fileURL, options: [.mappedIfSafe])
                let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                if hash.lowercased() != entry.sha256.lowercased() {
                    throw ModelPackageImportError.fileHashMismatch(path: entry.path)
                }
            }
        }.value
    }

    private func compareVersionStrings(_ lhs: String, _ rhs: String) -> ComparisonResult {
        let lhsParts = lhs.split(separator: ".").map { Int($0) ?? 0 }
        let rhsParts = rhs.split(separator: ".").map { Int($0) ?? 0 }
        let maxCount = max(lhsParts.count, rhsParts.count)

        for i in 0..<maxCount {
            let l = i < lhsParts.count ? lhsParts[i] : 0
            let r = i < rhsParts.count ? rhsParts[i] : 0
            if l < r { return .orderedAscending }
            if l > r { return .orderedDescending }
        }
        return .orderedSame
    }

    private func shouldProcessIncomingTransferURL(_ url: URL, dedupeWindow: TimeInterval = 10) -> Bool {
        let now = Date()
        recentlyHandledIncomingURLKeys = recentlyHandledIncomingURLKeys.filter { now.timeIntervalSince($0.value) <= dedupeWindow }
        let key = url.standardizedFileURL.path
        if recentlyHandledIncomingURLKeys[key] != nil {
            return false
        }
        recentlyHandledIncomingURLKeys[key] = now
        return true
    }

    private func cleanupIncomingPackageArtifactsIfNeeded(incomingURL: URL, resolvedPackageRoot: URL) {
        let fileManager = FileManager.default
        let documentsRoot = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0].standardizedFileURL
        #if os(macOS)
        // macOS: models/conversations under ~/.cache/anemll/
        let appDataRoot = fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache", isDirectory: true)
            .appendingPathComponent("anemll", isDirectory: true)
            .standardizedFileURL
        let modelsRoot = appDataRoot.appendingPathComponent("Models", isDirectory: true).standardizedFileURL
        let conversationsRoot = appDataRoot.appendingPathComponent("Conversations", isDirectory: true).standardizedFileURL
        #else
        // iOS: models/conversations under Documents/
        let modelsRoot = documentsRoot.appendingPathComponent("Models", isDirectory: true).standardizedFileURL
        let conversationsRoot = documentsRoot.appendingPathComponent("Conversations", isDirectory: true).standardizedFileURL
        #endif
        let inboxRoot = documentsRoot
            .appendingPathComponent("Inbox", isDirectory: true)
            .standardizedFileURL
        let tempRoot = fileManager.temporaryDirectory.standardizedFileURL
        let cachesRoot = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0].standardizedFileURL

        let candidates = [incomingURL.standardizedFileURL, resolvedPackageRoot.standardizedFileURL]
        var seen = Set<String>()
        for candidate in candidates where seen.insert(candidate.path).inserted {
            guard shouldDeleteIncomingPackageArtifact(
                candidate,
                inboxRoot: inboxRoot,
                tempRoot: tempRoot,
                cachesRoot: cachesRoot,
                modelsRoot: modelsRoot,
                conversationsRoot: conversationsRoot
            ) else {
                continue
            }
            guard fileManager.fileExists(atPath: candidate.path) else { continue }
            if removeItemBestEffort(candidate, fileManager: fileManager) {
                logInfo("Cleaned up imported transfer artifact: \(candidate.lastPathComponent)", category: .model)
            }
        }
    }

    private func shouldDeleteIncomingPackageArtifact(
        _ url: URL,
        inboxRoot: URL,
        tempRoot: URL,
        cachesRoot: URL,
        modelsRoot: URL,
        conversationsRoot: URL
    ) -> Bool {
        let normalized = url.standardizedFileURL.path
        if isPath(normalized, inside: modelsRoot.path) || isPath(normalized, inside: conversationsRoot.path) {
            return false
        }

        if url.pathExtension.lowercased() == "anemllpkg" {
            return true
        }

        return isPath(normalized, inside: inboxRoot.path)
            || isPath(normalized, inside: tempRoot.path)
            || isPath(normalized, inside: cachesRoot.path)
    }

    private func removeItemBestEffort(_ url: URL, fileManager: FileManager) -> Bool {
        var coordinationError: NSError?
        var removed = false

        let coordinator = NSFileCoordinator()
        coordinator.coordinate(writingItemAt: url, options: .forDeleting, error: &coordinationError) { coordinatedURL in
            do {
                if fileManager.fileExists(atPath: coordinatedURL.path) {
                    try fileManager.removeItem(at: coordinatedURL)
                }
                removed = true
            } catch {
                logWarning("Failed coordinated cleanup for \(coordinatedURL.path): \(error)", category: .model)
            }
        }

        if removed {
            return true
        }
        if let coordinationError {
            logWarning("File coordination failed for cleanup \(url.path): \(coordinationError)", category: .model)
        }

        do {
            if fileManager.fileExists(atPath: url.path) {
                try fileManager.removeItem(at: url)
            }
            return true
        } catch {
            logWarning("Failed to clean up transfer artifact \(url.path): \(error)", category: .model)
            return false
        }
    }

    private func isPath(_ child: String, inside parent: String) -> Bool {
        if child == parent { return true }
        return child.hasPrefix(parent.hasSuffix("/") ? parent : parent + "/")
    }

    // MARK: - Directory Size

    /// Calculate the total size of a directory in bytes
    func calculateDirectorySize(at url: URL) -> Int64 {
        let fileManager = FileManager.default
        var totalSize: Int64 = 0
        if let contents = try? fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey]) {
            for item in contents {
                if let values = try? item.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey]) {
                    if values.isDirectory == true {
                        totalSize += calculateDirectorySize(at: item)
                    } else if let fileSize = values.fileSize {
                        totalSize += Int64(fileSize)
                    }
                }
            }
        }
        return totalSize
    }

    /// Get formatted size string for a model directory
    func formattedModelSize(for model: ModelInfo) -> String? {
        return withLinkedModelAccess(for: model) { url in
            let bytes = calculateDirectorySize(at: url)
            guard bytes > 0 else { return nil }
            return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
        } ?? nil
    }

    // MARK: - Package Share (macOS sender)

    #if os(macOS)
    func shareModelToIOS(_ model: ModelInfo) async {
        isSharingModel = true
        sharingModelId = model.id

        do {
            guard model.isDownloaded, let localPath = model.localPath else {
                throw LocalModelValidationError.invalidStructure(["Model files are unavailable."])
            }

            let sourceRoot = URL(fileURLWithPath: localPath, isDirectory: true)
            let packageURL = try await buildTransferPackage(for: model, sourceRoot: sourceRoot)
            preparedTransferPackages.append(packageURL)
            scheduleTransferPackageCleanup(packageURL)

            isSharingModel = false
            sharingModelId = nil

            if let airDrop = NSSharingService(named: .sendViaAirDrop) {
                airDrop.perform(withItems: [packageURL])
                logInfo("Started AirDrop share for model package: \(packageURL.lastPathComponent)", category: .model)
            } else {
                NSWorkspace.shared.activateFileViewerSelecting([packageURL])
                errorMessage = "AirDrop service unavailable. Opened package in Finder."
            }
        } catch {
            isSharingModel = false
            sharingModelId = nil
            errorMessage = error.localizedDescription
            logError("Failed to share model package: \(error)", category: .model)
        }
    }

    private func buildTransferPackage(for model: ModelInfo, sourceRoot: URL) async throws -> URL {
        let validatedRoot = try detectLocalModelRoot(from: sourceRoot)
        let fileManager = FileManager.default
        cleanupStaleTransferPackages()
        let now = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let safeModelName = slugify(model.name)

        let packageRoot = fileManager.temporaryDirectory
            .appendingPathComponent("anemll-transfer", isDirectory: true)
            .appendingPathComponent("\(safeModelName)-\(now).anemllpkg", isDirectory: true)
        let modelOut = packageRoot.appendingPathComponent("model", isDirectory: true)

        if fileManager.fileExists(atPath: packageRoot.path) {
            try fileManager.removeItem(at: packageRoot)
        }

        try fileManager.createDirectory(at: packageRoot, withIntermediateDirectories: true)
        try fileManager.copyItem(at: validatedRoot, to: modelOut)

        let fileEntries = try buildManifestEntries(forModelRoot: modelOut, packageRootPathPrefix: "model")
        let appVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
        let manifest = ModelPackageManifest(
            formatVersion: 1,
            modelName: model.name,
            modelId: model.id,
            modelRootPath: "model",
            minAppVersion: appVersion,
            files: fileEntries
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let manifestData = try encoder.encode(manifest)
        try manifestData.write(to: packageRoot.appendingPathComponent("manifest.json"), options: .atomic)

        return packageRoot
    }

    private func buildManifestEntries(forModelRoot modelRoot: URL, packageRootPathPrefix: String) throws -> [ModelPackageFileEntry] {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: modelRoot,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        guard let normalizedPrefix = Self.normalizePackageRelativePath(packageRootPathPrefix) else {
            return []
        }

        var entries: [ModelPackageFileEntry] = []
        for case let fileURL as URL in enumerator {
            let values = try fileURL.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
            if values.isDirectory == true {
                continue
            }

            guard let relative = relativePath(from: modelRoot, to: fileURL),
                  let normalizedRelative = Self.normalizePackageRelativePath(relative) else {
                logWarning("Skipping file with invalid relative path in transfer package: \(fileURL.path)", category: .model)
                continue
            }

            let manifestPath = "\(normalizedPrefix)/\(normalizedRelative)"
            let sha = try sha256Hex(for: fileURL)
            entries.append(ModelPackageFileEntry(path: manifestPath, sha256: sha, sizeBytes: values.fileSize.map(Int64.init)))
        }

        return entries.sorted { $0.path < $1.path }
    }

    private func cleanupStaleTransferPackages(maxAge: TimeInterval = 24 * 60 * 60) {
        let fileManager = FileManager.default
        let transferRoot = fileManager.temporaryDirectory.appendingPathComponent("anemll-transfer", isDirectory: true)
        guard let entries = try? fileManager.contentsOfDirectory(
            at: transferRoot,
            includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            preparedTransferPackages.removeAll { !fileManager.fileExists(atPath: $0.path) }
            return
        }

        let cutoff = Date().addingTimeInterval(-maxAge)
        for entry in entries where entry.pathExtension.lowercased() == "anemllpkg" {
            let values = try? entry.resourceValues(forKeys: [.isDirectoryKey, .contentModificationDateKey])
            guard values?.isDirectory == true else { continue }
            if let modified = values?.contentModificationDate, modified < cutoff {
                try? fileManager.removeItem(at: entry)
            }
        }

        preparedTransferPackages.removeAll { !fileManager.fileExists(atPath: $0.path) }
    }

    private func scheduleTransferPackageCleanup(_ packageURL: URL, delaySeconds: TimeInterval = 45 * 60) {
        let packagePath = packageURL.standardizedFileURL.path
        Task { @MainActor [weak self] in
            let delay = max(delaySeconds, 0)
            let nanoseconds = UInt64(delay * 1_000_000_000)
            try? await Task.sleep(nanoseconds: nanoseconds)

            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: packagePath) {
                try? fileManager.removeItem(atPath: packagePath)
            }

            self?.preparedTransferPackages.removeAll {
                let trackedPath = $0.standardizedFileURL.path
                return trackedPath == packagePath || !fileManager.fileExists(atPath: trackedPath)
            }
        }
    }

    private func relativePath(from root: URL, to file: URL) -> String? {
        let rootPath = root.resolvingSymlinksInPath().standardizedFileURL.path
        let filePath = file.resolvingSymlinksInPath().standardizedFileURL.path

        if filePath == rootPath {
            return ""
        }
        if filePath.hasPrefix(rootPath + "/") {
            return String(filePath.dropFirst(rootPath.count + 1))
        }

        // Fallback for path representation mismatches.
        let rootComponents = root.standardizedFileURL.pathComponents
        let fileComponents = file.standardizedFileURL.pathComponents
        guard fileComponents.count >= rootComponents.count else {
            return nil
        }
        guard Array(fileComponents.prefix(rootComponents.count)) == rootComponents else {
            return nil
        }

        return fileComponents.dropFirst(rootComponents.count).joined(separator: "/")
    }

    private func sha256Hex(for fileURL: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer {
            try? handle.close()
        }

        var hasher = SHA256()
        while true {
            let data = try handle.read(upToCount: 1_048_576) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }
    #endif

    /// Internal download implementation (called after downloadingModelId is set)
    private func performDownload(_ model: ModelInfo) async {
        logInfo("Starting download: \(model.id)", category: .download)

        await DownloadService.shared.downloadModel(
            model.id,
            progress: { [weak self] progress in
                Task { @MainActor in
                    self?.downloadProgress = progress
                    self?.updateModelProgress(model.id, progress: progress)
                }
            },
            completion: { [weak self] result in
                Task { @MainActor in
                    guard let self = self else { return }

                    switch result {
                    case .success(let path):
                        self.updateModelDownloaded(model.id, path: path)
                        logInfo("Download complete: \(model.id)", category: .download)

                    case .failure(let error):
                        self.updateModelError(model.id, error: error.localizedDescription)
                        self.errorMessage = error.localizedDescription
                        logError("Download failed: \(error)", category: .download)
                    }

                    self.downloadingModelId = nil
                    self.downloadProgress = nil
                }
            }
        )
    }

    // (HuggingFace author-search discovery removed — refresh now uses the curated collection API)

    // MARK: - Helpers

    nonisolated private static func normalizePackageRelativePath(_ rawPath: String) -> String? {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let unified = trimmed.replacingOccurrences(of: "\\", with: "/")
        let components = unified.split(separator: "/", omittingEmptySubsequences: true)
        if components.isEmpty {
            return nil
        }

        var cleaned: [String] = []
        cleaned.reserveCapacity(components.count)

        for raw in components {
            let part = String(raw)
            if part == "." {
                continue
            }
            if part == ".." {
                return nil
            }
            cleaned.append(part)
        }

        guard !cleaned.isEmpty else { return nil }
        return cleaned.joined(separator: "/")
    }

    nonisolated private static func resolvePackageFileURL(
        for normalizedPath: String,
        packageRoot: URL,
        fileManager: FileManager
    ) -> URL? {
        var candidates: [String] = [normalizedPath]
        candidates.append(contentsOf: legacyPathCandidates(for: normalizedPath))

        var seen = Set<String>()
        for candidate in candidates where seen.insert(candidate).inserted {
            let url = packageRoot.appendingPathComponent(candidate)
            if fileManager.fileExists(atPath: url.path) {
                return url
            }
        }
        return nil
    }

    nonisolated private static func legacyPathCandidates(for normalizedPath: String) -> [String] {
        let components = normalizedPath.split(separator: "/").map(String.init)
        guard components.count >= 2 else { return [] }

        let root = components[0]
        var candidates: [String] = []

        // Compatibility for older sender bug that produced model//private*.*
        if components[1].hasPrefix("private"), components[1].count > "private".count {
            var fixed = components
            fixed[1] = String(fixed[1].dropFirst("private".count))
            if !fixed[1].isEmpty {
                candidates.append(fixed.joined(separator: "/"))
            }
        }

        // Compatibility when an absolute path fragment leaked into manifest and contains /model/... twice.
        let marker = "/\(root)/"
        if let range = normalizedPath.range(of: marker, options: .backwards) {
            let suffix = String(normalizedPath[range.upperBound...])
            if !suffix.isEmpty {
                candidates.append("\(root)/\(suffix)")
            }
        }

        return candidates
    }

    private func refreshedModelStatus(for inputModel: ModelInfo) async -> ModelInfo {
        var model = inputModel
        // Reset transient download state (downloads don't survive app restart)
        model.isDownloading = false
        model.downloadProgress = nil

        switch model.sourceKind {
        case .huggingFace, .localImported:
            let isDownloaded = await StorageService.shared.isModelDownloaded(model.id)
            model.isDownloaded = isDownloaded

            if isDownloaded {
                let path = await StorageService.shared.modelPath(for: model.id)
                model.localPath = path.path
                model.metaYamlPath = path.appendingPathComponent("meta.yaml").path
                model.downloadError = nil
                logDebug("Model \(model.name) is available at \(path.path)", category: .model)
            } else if model.sourceKind == .localImported {
                model.downloadError = "Imported model files are missing."
            } else {
                model.downloadError = nil
            }

        case .localLinked:
            logDebug("Checking linked model: \(model.id) path=\(model.linkedPath ?? model.localPath ?? "nil")", category: .model)

            // Capture Sendable values for the off-main-actor task.
            // Both bookmark resolution and file validation can block for 30+ seconds
            // on unreachable network volumes, so everything runs inside a timeout.
            let bookmarkBase64 = model.bookmarkDataBase64
            let fallbackPath = model.linkedPath ?? model.localPath

            let linkedCheckResult: (path: String, issues: [String])? = await withTaskGroup(
                of: (String, [String])?.self
            ) { group in
                group.addTask(priority: .userInitiated) {
                    // Step 1: Resolve bookmark (this can block on stale network mounts)
                    var resolvedPath: String? = nil
                    #if os(macOS)
                    if let base64 = bookmarkBase64, let bookmarkData = Data(base64Encoded: base64) {
                        var isStale = false
                        if let url = try? URL(
                            resolvingBookmarkData: bookmarkData,
                            options: [.withSecurityScope, .withoutUI],
                            relativeTo: nil,
                            bookmarkDataIsStale: &isStale
                        ) {
                            _ = url.startAccessingSecurityScopedResource()
                            resolvedPath = url.path
                        } else if let url = try? URL(
                            resolvingBookmarkData: bookmarkData,
                            options: [.withoutUI],
                            relativeTo: nil,
                            bookmarkDataIsStale: &isStale
                        ) {
                            resolvedPath = url.path
                        }
                    }
                    #endif
                    let finalPath = resolvedPath ?? fallbackPath
                    guard let finalPath else { return nil }

                    // Step 2: Validate the model directory
                    let pathURL = URL(fileURLWithPath: finalPath)
                    let fm = FileManager.default
                    var issues: [String] = []
                    var isDirectory: ObjCBool = false
                    guard fm.fileExists(atPath: pathURL.path, isDirectory: &isDirectory),
                          isDirectory.boolValue else {
                        return (finalPath, ["not a folder"])
                    }
                    if !fm.fileExists(atPath: pathURL.appendingPathComponent("meta.yaml").path) {
                        issues.append("meta.yaml")
                    }
                    let hasMLModelc = ((try? fm.contentsOfDirectory(at: pathURL, includingPropertiesForKeys: nil)) ?? [])
                        .contains { $0.pathExtension.lowercased() == "mlmodelc" }
                    if !hasMLModelc {
                        issues.append("*.mlmodelc")
                    }
                    return (finalPath, issues)
                }
                group.addTask {
                    try? await Task.sleep(for: .seconds(3))
                    return nil  // timeout sentinel
                }
                // First to finish wins
                for await result in group {
                    group.cancelAll()
                    return result
                }
                return nil
            }

            if let result = linkedCheckResult {
                let linkedRoot = URL(fileURLWithPath: result.path)
                model.localPath = linkedRoot.path
                model.linkedPath = linkedRoot.path
                model.metaYamlPath = linkedRoot.appendingPathComponent("meta.yaml").path
                model.isDownloaded = result.issues.isEmpty
                model.downloadError = result.issues.isEmpty ? nil : "Linked source folder missing: \(result.issues.joined(separator: ", "))"
            } else {
                // Timed out — bookmark resolution or file access hung on unreachable network volume
                logWarning("Linked model \(model.id) path check timed out (network unreachable?): \(fallbackPath ?? "nil")", category: .model)
                model.isDownloaded = false
                model.downloadError = "Linked source folder is unreachable (timed out). Remove and re-link when the volume is mounted."
            }
        }

        return model
    }

    private func detectLocalModelRoot(from droppedURL: URL) throws -> URL {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: droppedURL.path, isDirectory: &isDirectory) else {
            throw LocalModelValidationError.invalidStructure([
                "Folder does not exist at path \(droppedURL.path)."
            ])
        }
        guard isDirectory.boolValue else {
            throw LocalModelValidationError.notDirectory
        }

        let startURL = droppedURL.standardizedFileURL
        var candidates: [URL] = []
        candidates.append(startURL)
        candidates.append(startURL.appendingPathComponent("ios", isDirectory: true))
        candidates.append(startURL.appendingPathComponent("hf", isDirectory: true).appendingPathComponent("ios", isDirectory: true))

        // Check parents too in case user drops a containing folder above the model root.
        var parent = startURL.deletingLastPathComponent()
        for _ in 0..<3 {
            candidates.append(parent)
            candidates.append(parent.appendingPathComponent("hf", isDirectory: true).appendingPathComponent("ios", isDirectory: true))
            let nextParent = parent.deletingLastPathComponent()
            if nextParent == parent { break }
            parent = nextParent
        }

        for candidate in deduplicatedURLs(candidates) {
            if validationIssues(forModelRoot: candidate).isEmpty {
                return candidate
            }
        }

        if let descendant = findDescendantModelRoot(from: startURL, maxDepth: 3) {
            return descendant
        }

        throw LocalModelValidationError.invalidStructure([
            "Expected a model folder containing meta.yaml and at least one .mlmodelc directory.",
            "Dropped path: \(droppedURL.path)"
        ])
    }

    private func findDescendantModelRoot(from startURL: URL, maxDepth: Int) -> URL? {
        let fileManager = FileManager.default
        var queue: [(url: URL, depth: Int)] = [(startURL, 0)]
        var visited = Set<String>()

        while !queue.isEmpty {
            let (currentURL, depth) = queue.removeFirst()
            if visited.contains(currentURL.path) { continue }
            visited.insert(currentURL.path)

            if validationIssues(forModelRoot: currentURL).isEmpty {
                return currentURL
            }

            guard depth < maxDepth else { continue }
            guard let children = try? fileManager.contentsOfDirectory(
                at: currentURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }

            for child in children {
                if let values = try? child.resourceValues(forKeys: [.isDirectoryKey]), values.isDirectory == true {
                    queue.append((child, depth + 1))
                }
            }
        }

        return nil
    }

    private func validationIssues(forModelRoot url: URL) -> [String] {
        let fileManager = FileManager.default
        var issues: [String] = []

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return ["not a folder"]
        }

        let metaYaml = url.appendingPathComponent("meta.yaml")
        if !fileManager.fileExists(atPath: metaYaml.path) {
            issues.append("meta.yaml")
        }

        let hasMLModelc: Bool = ((try? fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? [])
            .contains(where: { $0.pathExtension.lowercased() == "mlmodelc" })
        if !hasMLModelc {
            issues.append("*.mlmodelc")
        }

        return issues
    }

    private func suggestModelName(from droppedURL: URL) -> String {
        let components = droppedURL.standardizedFileURL.pathComponents.filter { $0 != "/" && !$0.isEmpty }
        if components.count >= 3 {
            let leaf = components[components.count - 1]
            let parent = components[components.count - 2]

            // Prefer grandparent for common converter/export layouts like:
            // .../<name>/hf/ios and .../<name>/hf_dist/ios
            if isIOSFolderName(leaf) && isHFExportFolderName(parent) {
                return beautifyDisplayName(components[components.count - 3])
            }
        }

        for component in components.reversed() {
            if !isGenericModelFolderName(component) {
                return beautifyDisplayName(component)
            }
        }

        return "Model"
    }

    private func beautifyDisplayName(_ raw: String) -> String {
        let cleaned = raw
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty ? "Model" : cleaned
    }

    private func isIOSFolderName(_ value: String) -> Bool {
        normalizedFolderToken(value) == "ios"
    }

    private func isHFExportFolderName(_ value: String) -> Bool {
        let lower = value.lowercased()
        let normalized = normalizedFolderToken(value)

        if lower == "hf" || lower == "huggingface" {
            return true
        }
        if lower.hasPrefix("hf_") || lower.hasPrefix("hf-") {
            return true
        }
        if normalized.hasPrefix("hf") && normalized.hasSuffix("dist") {
            return true
        }
        return false
    }

    private func isGenericModelFolderName(_ value: String) -> Bool {
        let lower = value.lowercased()
        let normalized = normalizedFolderToken(value)
        let genericExact: Set<String> = [
            "ios", "hf", "huggingface",
            "model", "models",
            "output", "outputs",
            "converted", "convert",
            "dist", "build"
        ]

        if genericExact.contains(lower) || genericExact.contains(normalized) {
            return true
        }
        if isHFExportFolderName(lower) {
            return true
        }
        return false
    }

    private func normalizedFolderToken(_ value: String) -> String {
        value
            .lowercased()
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: " ", with: "")
    }

    private func uniqueDisplayName(for requestedName: String) -> String {
        let base = beautifyDisplayName(requestedName)
        let existing = Set(availableModels.map { $0.name.lowercased() })
        if !existing.contains(base.lowercased()) {
            return base
        }

        var suffix = 2
        while true {
            let candidate = "\(base)-\(suffix)"
            if !existing.contains(candidate.lowercased()) {
                return candidate
            }
            suffix += 1
        }
    }

    private func uniqueLocalModelId(forDisplayName displayName: String) -> String {
        let baseSlug = slugify(displayName)
        var candidate = "local/\(baseSlug)"
        let existingIds = Set(availableModels.map { $0.id })
        if !existingIds.contains(candidate) {
            return candidate
        }

        var suffix = 2
        while true {
            candidate = "local/\(baseSlug)-\(suffix)"
            if !existingIds.contains(candidate) {
                return candidate
            }
            suffix += 1
        }
    }

    private func slugify(_ value: String) -> String {
        let lower = value.lowercased()
        let allowed = CharacterSet.alphanumerics
        var buffer = ""
        var previousWasHyphen = false

        for scalar in lower.unicodeScalars {
            if allowed.contains(scalar) {
                buffer.append(Character(scalar))
                previousWasHyphen = false
            } else if !previousWasHyphen {
                buffer.append("-")
                previousWasHyphen = true
            }
        }

        let trimmed = buffer.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return trimmed.isEmpty ? "model" : trimmed
    }

    private func deduplicatedURLs(_ urls: [URL]) -> [URL] {
        var seen = Set<String>()
        var result: [URL] = []
        for url in urls {
            if seen.insert(url.path).inserted {
                result.append(url)
            }
        }
        return result
    }

    private func resolveLinkedModelPath(for model: ModelInfo) -> String? {
        return resolveLinkedModelURL(for: model)?.path
    }

    /// Resolve a linked model's bookmark into a security-scoped URL.
    /// The caller must call `stopAccessingSecurityScopedResource()` when done.
    private func resolveLinkedModelURL(for model: ModelInfo) -> URL? {
        #if os(macOS)
        if let base64 = model.bookmarkDataBase64, let bookmarkData = Data(base64Encoded: base64) {
            var isStale = false
            // Try security-scoped resolution first, then fall back to non-scoped
            if let resolvedURL = try? URL(
                resolvingBookmarkData: bookmarkData,
                options: [.withSecurityScope, .withoutUI],
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            ) {
                _ = resolvedURL.startAccessingSecurityScopedResource()
                return resolvedURL
            }
            if let resolvedURL = try? URL(
                resolvingBookmarkData: bookmarkData,
                options: [.withoutUI],
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            ) {
                return resolvedURL
            }
        }
        #endif
        if let path = model.linkedPath ?? model.localPath {
            return URL(fileURLWithPath: path)
        }
        return nil
    }

    /// Execute a closure with security-scoped access to a linked model's folder.
    /// For non-linked models, runs the closure with the local path directly.
    /// Automatically starts/stops security-scoped resource access.
    func withLinkedModelAccess<T>(for model: ModelInfo, body: (URL) -> T) -> T? {
        if model.sourceKind == .localLinked {
            guard let url = resolveLinkedModelURL(for: model) else { return nil }
            defer { url.stopAccessingSecurityScopedResource() }
            return body(url)
        } else {
            guard let path = model.localPath else { return nil }
            return body(URL(fileURLWithPath: path))
        }
    }

    /// Async version of withLinkedModelAccess.
    func withLinkedModelAccess<T>(for model: ModelInfo, body: (URL) async -> T) async -> T? {
        if model.sourceKind == .localLinked {
            guard let url = resolveLinkedModelURL(for: model) else { return nil }
            let result = await body(url)
            url.stopAccessingSecurityScopedResource()
            return result
        } else {
            guard let path = model.localPath else { return nil }
            return await body(URL(fileURLWithPath: path))
        }
    }

    /// Async throwing version of withLinkedModelAccess.
    func withLinkedModelAccess<T>(for model: ModelInfo, body: (URL) async throws -> T) async throws -> T? {
        if model.sourceKind == .localLinked {
            guard let url = resolveLinkedModelURL(for: model) else { return nil }
            do {
                let result = try await body(url)
                url.stopAccessingSecurityScopedResource()
                return result
            } catch {
                url.stopAccessingSecurityScopedResource()
                throw error
            }
        } else {
            guard let path = model.localPath else { return nil }
            return try await body(URL(fileURLWithPath: path))
        }
    }

    private func markModelAsJustAdded(_ id: String) {
        justAddedModelId = id
        clearJustAddedTask?.cancel()
        clearJustAddedTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(90))
            guard let self else { return }
            if self.justAddedModelId == id {
                self.justAddedModelId = nil
            }
        }
    }

    private func makeBookmarkDataBase64(for url: URL) -> String? {
        #if os(macOS)
        do {
            // Try security-scoped bookmark first (works in sandboxed apps)
            let bookmark = try url.bookmarkData(options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil)
            return bookmark.base64EncodedString()
        } catch {
            // Fall back to non-scoped bookmark (works without sandbox)
            do {
                let bookmark = try url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
                return bookmark.base64EncodedString()
            } catch {
                logWarning("Failed to create bookmark: \(error)", category: .model)
                return nil
            }
        }
        #else
        return nil
        #endif
    }

    private func updateModelDownloading(_ id: String, isDownloading: Bool) {
        if let index = availableModels.firstIndex(where: { $0.id == id }) {
            availableModels[index].isDownloading = isDownloading
            if !isDownloading {
                availableModels[index].downloadProgress = nil
                if availableModels[index].sourceKind == .huggingFace {
                    availableModels[index].downloadError = nil
                }
            }
        }
    }

    private func updateModelProgress(_ id: String, progress: DownloadProgress) {
        if let index = availableModels.firstIndex(where: { $0.id == id }) {
            availableModels[index].downloadProgress = progress.progress
            availableModels[index].downloadedBytes = progress.downloadedBytes
        }
    }

    private func updateModelDownloaded(_ id: String, path: URL) {
        if let index = availableModels.firstIndex(where: { $0.id == id }) {
            availableModels[index].isDownloaded = true
            availableModels[index].isDownloading = false
            availableModels[index].localPath = path.path
            availableModels[index].metaYamlPath = path.appendingPathComponent("meta.yaml").path
            availableModels[index].downloadProgress = nil
            availableModels[index].downloadError = nil

            // Parse meta.yaml immediately after download to evaluate M1/A14 compatibility.
            let warning = evaluateGlobalAttentionWarning(for: availableModels[index])
            cacheGlobalAttentionWarning(warning, for: id)
            if let warning {
                logWarning("Post-download compatibility warning for \(id): \(warning)", category: .model)
            } else {
                logDebug("Post-download compatibility check passed: \(id)", category: .model)
            }
        }
    }

    private func updateModelError(_ id: String, error: String) {
        if let index = availableModels.firstIndex(where: { $0.id == id }) {
            availableModels[index].isDownloading = false
            availableModels[index].downloadError = error
        }
        clearCachedGlobalAttentionWarning(for: id)
    }

    // MARK: - Weight File Size Checking

    /// Get detailed weight file information for a model
    /// - Parameter model: The model to check
    /// - Returns: Tuple with (largestWeightSize, largestWeightName, allWeightFiles) or nil if not available
    func getWeightFileDetails(for model: ModelInfo) -> (largest: Int64, largestName: String, files: [(name: String, size: Int64)])? {
        return withLinkedModelAccess(for: model) { modelDir in
            let fileManager = FileManager.default

            guard fileManager.fileExists(atPath: modelDir.path) else { return nil }

            var weightFiles: [(name: String, size: Int64)] = []

            // Check all .mlmodelc directories for weight.bin files
            do {
                let contents = try fileManager.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
                let mlmodelcDirs = contents.filter { $0.pathExtension == "mlmodelc" }

                for mlmodelcDir in mlmodelcDirs {
                    let dirName = mlmodelcDir.lastPathComponent

                    // Check both possible weight file locations
                    let weightPaths = [
                        mlmodelcDir.appendingPathComponent("weights/weight.bin"),
                        mlmodelcDir.appendingPathComponent("weight.bin")
                    ]

                    for weightPath in weightPaths {
                        if fileManager.fileExists(atPath: weightPath.path) {
                            if let attrs = try? fileManager.attributesOfItem(atPath: weightPath.path),
                               let size = attrs[.size] as? Int64, size > 0 {
                                weightFiles.append((name: dirName, size: size))
                            }
                            break
                        }
                    }
                }
            } catch {
                logError("Error getting weight file details: \(error)", category: .model)
                return nil
            }

            guard !weightFiles.isEmpty else { return nil }

            let sorted = weightFiles.sorted { $0.size > $1.size }
            let largest = sorted.first!

            return (largest: largest.size, largestName: largest.name, files: weightFiles)
        } ?? nil
    }

    /// Get warning message if weight files exceed 1GB on limited devices
    /// - Parameter model: The model to check
    /// - Returns: Warning message or nil if no issue
    func getWeightSizeWarning(for model: ModelInfo) -> String? {
        guard DeviceType.requiresWeightSizeLimit else { return nil }
        guard let details = getWeightFileDetails(for: model) else { return nil }

        let oversizedFiles = details.files.filter { $0.size > DeviceType.maxWeightFileSize }
        guard !oversizedFiles.isEmpty else { return nil }

        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file

        let fileList = oversizedFiles.map { "\($0.name): \(formatter.string(fromByteCount: $0.size))" }.joined(separator: ", ")
        let deviceName: String
        switch DeviceType.current {
        case .iPhone:
            deviceName = "iPhone"
        case .iPad:
            deviceName = "this iPad (non-M-series)"
        default:
            deviceName = "this device"
        }

        return "Model may not load on \(deviceName). Weight file(s) exceed 1GB limit: \(fileList)"
    }

    /// Check if model has any weight files exceeding 1GB (regardless of device)
    /// - Parameter model: The model to check
    /// - Returns: True if any weight file exceeds 1GB
    func hasOversizedWeights(for model: ModelInfo) -> Bool {
        guard let details = getWeightFileDetails(for: model) else { return false }
        return details.largest > DeviceType.maxWeightFileSize
    }

    // MARK: - ANE Compatibility Checking

    // [ANE-COMPAT:M1-A14] Check model compatibility with device
    /// Check if a model uses global attention that is incompatible with this device.
    /// Works for both downloaded models (reads meta.yaml from disk) and pre-checked models
    /// (returns cached result from pre-download meta.yaml fetch).
    /// Returns a warning message, or nil if compatible.
    /// NOTE: Currently only Gemma models are affected. Other architectures (Qwen, LLaMA)
    /// may report sliding_window but don't have the same KV-cache rotation issue.
    func getGlobalAttentionWarning(for model: ModelInfo) -> String? {
        guard !DeviceType.supportsGlobalAttention else { return nil }

        // Return cached result if available (covers both post-download and pre-download checks)
        if globalAttentionCheckedModelIds.contains(model.id) {
            return globalAttentionWarningByModelId[model.id]
        }

        // For downloaded models, evaluate from local meta.yaml
        guard model.isDownloaded else { return nil }

        let warning = evaluateGlobalAttentionWarning(for: model)
        cacheGlobalAttentionWarning(warning, for: model.id)
        return warning
    }

    private func evaluateGlobalAttentionWarning(for model: ModelInfo) -> String? {
        guard !DeviceType.supportsGlobalAttention else { return nil }

        // Resolve bookmark for linked models to read meta.yaml
        let metadata: ModelMetadata? = withLinkedModelAccess(for: model) { modelURL in
            let metaPath = modelURL.appendingPathComponent("meta.yaml").path
            return ModelMetadata.load(from: metaPath)
        } ?? nil
        guard let metadata else { return nil }

        // Only Gemma architecture is affected by this limitation.
        let isGemma = metadata.isGemmaFamily
            || model.architecture?.lowercased() == "gemma"
            || model.id.lowercased().contains("gemma")
        guard isGemma else { return nil }

        // When contextLength > slidingWindow, the model uses global attention layers
        // beyond the SWA range — this requires M2+/A15+ hardware
        guard let sw = metadata.slidingWindow, metadata.contextLength > sw else { return nil }
        return "This Gemma model uses global attention (context \(metadata.contextLength) > SWA \(sw)) which is not compatible with \(DeviceType.chipName). Requires Apple M2+ (Mac) or A15+ (iPhone/iPad)."
    }

    private func cacheGlobalAttentionWarning(_ warning: String?, for modelId: String) {
        globalAttentionCheckedModelIds.insert(modelId)
        if let warning {
            globalAttentionWarningByModelId[modelId] = warning
        } else {
            globalAttentionWarningByModelId.removeValue(forKey: modelId)
        }
    }

    private func clearCachedGlobalAttentionWarning(for modelId: String) {
        globalAttentionCheckedModelIds.remove(modelId)
        globalAttentionWarningByModelId.removeValue(forKey: modelId)
    }

    private func refreshGlobalAttentionCompatibilityCache() {
        globalAttentionWarningByModelId.removeAll()
        globalAttentionCheckedModelIds.removeAll()

        guard !DeviceType.supportsGlobalAttention else { return }
        for model in availableModels where model.isDownloaded {
            let warning = evaluateGlobalAttentionWarning(for: model)
            cacheGlobalAttentionWarning(warning, for: model.id)
        }
    }
}
