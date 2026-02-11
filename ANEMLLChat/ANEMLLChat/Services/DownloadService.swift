//
//  DownloadService.swift
//  ANEMLLChat
//
//  HuggingFace model download management
//

import Foundation

/// Errors during model download
enum DownloadError: LocalizedError {
    case invalidURL
    case networkError(Error)
    case fileListFailed
    case downloadFailed(String)
    case verificationFailed
    case cancelled

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid download URL"
        case .networkError(let error): return "Network error: \(error.localizedDescription)"
        case .fileListFailed: return "Failed to get file list from HuggingFace"
        case .downloadFailed(let file): return "Failed to download: \(file)"
        case .verificationFailed: return "Downloaded files failed verification"
        case .cancelled: return "Download was cancelled"
        }
    }
}

/// Progress information for downloads
struct DownloadProgress: Sendable {
    let totalBytes: Int64
    let downloadedBytes: Int64
    let currentFile: String
    let filesCompleted: Int
    let totalFiles: Int
    let bytesPerSecond: Double

    var progress: Double {
        totalBytes > 0 ? Double(downloadedBytes) / Double(totalBytes) : 0
    }

    var progressPercent: String {
        String(format: "%.0f%%", progress * 100)
    }

    var downloadedString: String {
        ByteCountFormatter.string(fromByteCount: downloadedBytes, countStyle: .file)
    }

    var totalString: String {
        ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file)
    }

    var speedString: String {
        ByteCountFormatter.string(fromByteCount: Int64(bytesPerSecond), countStyle: .file) + "/s"
    }

    var etaString: String? {
        guard bytesPerSecond > 0, totalBytes > downloadedBytes else { return nil }
        let remaining = Double(totalBytes - downloadedBytes)
        let seconds = remaining / bytesPerSecond

        if seconds < 60 {
            return String(format: "%.0fs", seconds)
        } else if seconds < 3600 {
            return String(format: "%.0fm", seconds / 60)
        } else {
            return String(format: "%.1fh", seconds / 3600)
        }
    }
}

/// Service for downloading models from HuggingFace
actor DownloadService: NSObject {
    static let shared = DownloadService()

    private var urlSession: URLSession!
    private var downloadTasks: [String: URLSessionDownloadTask] = [:]
    private var resumeData: [String: Data] = [:]
    private var progressCallbacks: [String: @Sendable (DownloadProgress) -> Void] = [:]
    private var completionCallbacks: [String: @Sendable (Result<URL, DownloadError>) -> Void] = [:]

    // Download state tracking
    private var downloadStartTime: [String: Date] = [:]
    private var downloadedBytesHistory: [String: [(Date, Int64)]] = [:]
    private var currentFileIndex: [String: Int] = [:]
    private var totalFilesCount: [String: Int] = [:]
    private var totalBytesExpected: [String: Int64] = [:]
    private var downloadedBytesTotal: [String: Int64] = [:]
    private var currentFileName: [String: String] = [:]
    private var progressObservations: [String: NSKeyValueObservation] = [:]
    private var downloadCompletions: [String: CheckedContinuation<Void, Error>] = [:]
    private var currentFileSize: [String: Int64] = [:]
    private var lastProgressUpdate: [String: Date] = [:]  // Throttle progress updates

    override private init() {
        super.init()

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 3600 // 1 hour for large files
        config.httpMaximumConnectionsPerHost = 3

        urlSession = URLSession(configuration: config, delegate: nil, delegateQueue: nil)
    }

    // MARK: - Public API

    /// Download a model from HuggingFace
    func downloadModel(
        _ modelId: String,
        progress: @escaping @Sendable (DownloadProgress) -> Void,
        completion: @escaping @Sendable (Result<URL, DownloadError>) -> Void
    ) async {
        logInfo("Starting download for model: \(modelId)", category: .download)

        progressCallbacks[modelId] = progress
        completionCallbacks[modelId] = completion
        downloadStartTime[modelId] = Date()

        do {
            // Get file list from HuggingFace
            let files = try await fetchFileList(for: modelId)
            logInfo("Found \(files.count) files to download", category: .download)

            totalFilesCount[modelId] = files.count
            totalBytesExpected[modelId] = files.reduce(0) { $0 + $1.size }
            downloadedBytesTotal[modelId] = 0

            // Create destination directory
            let destDir = await StorageService.shared.modelPath(for: modelId)
            try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

            // Send initial progress update
            let initialProgress = DownloadProgress(
                totalBytes: totalBytesExpected[modelId] ?? 0,
                downloadedBytes: 0,
                currentFile: files.first?.name ?? "",
                filesCompleted: 0,
                totalFiles: files.count,
                bytesPerSecond: 0
            )
            progress(initialProgress)

            // Download each file
            for (index, file) in files.enumerated() {
                currentFileIndex[modelId] = index
                currentFileName[modelId] = file.name

                try await downloadFile(file, to: destDir, modelId: modelId)

                // Update progress after each file completes
                let fileProgress = DownloadProgress(
                    totalBytes: totalBytesExpected[modelId] ?? 0,
                    downloadedBytes: downloadedBytesTotal[modelId] ?? 0,
                    currentFile: file.name,
                    filesCompleted: index + 1,
                    totalFiles: files.count,
                    bytesPerSecond: 0
                )
                progress(fileProgress)
            }

            // Verify download
            if await verifyDownload(modelId: modelId, at: destDir) {
                logInfo("Download complete and verified: \(modelId)", category: .download)
                completion(.success(destDir))
            } else {
                completion(.failure(.verificationFailed))
            }

        } catch let error as DownloadError {
            logError("Download failed: \(error)", category: .download)
            completion(.failure(error))
        } catch {
            logError("Download failed: \(error)", category: .download)
            completion(.failure(.networkError(error)))
        }

        cleanup(modelId: modelId)
    }

    /// Cancel an ongoing download
    func cancelDownload(_ modelId: String) {
        logInfo("Cancelling download: \(modelId)", category: .download)

        if let task = downloadTasks[modelId] {
            task.cancel { [weak self] data in
                Task {
                    await self?.storeResumeData(data, for: modelId)
                }
            }
        }

        completionCallbacks[modelId]?(.failure(.cancelled))
        cleanup(modelId: modelId)
    }

    /// Check if a download is in progress
    func isDownloading(_ modelId: String) -> Bool {
        downloadTasks[modelId] != nil
    }

    // MARK: - Pre-download Meta Fetch

    // [ANE-COMPAT:M1-A14] Fetch just meta.yaml from HuggingFace before starting full download
    /// Fetches the raw content of meta.yaml for a given model repo without downloading the full model.
    /// Returns the YAML string, or nil if the file doesn't exist or fetch fails.
    func fetchMetaYaml(for modelId: String) async -> String? {
        let urlString = "https://huggingface.co/\(modelId)/resolve/main/meta.yaml"
        guard let url = URL(string: urlString) else { return nil }

        do {
            let (data, response) = try await urlSession.data(from: url)
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                logDebug("meta.yaml not found for \(modelId) (status: \((response as? HTTPURLResponse)?.statusCode ?? 0))", category: .download)
                return nil
            }
            return String(data: data, encoding: .utf8)
        } catch {
            logDebug("Failed to fetch meta.yaml for \(modelId): \(error)", category: .download)
            return nil
        }
    }

    // MARK: - HuggingFace API

    private struct HFFile: Sendable {
        let name: String
        let url: URL
        let size: Int64
    }

    private func fetchFileList(for modelId: String) async throws -> [HFFile] {
        // Fetch root directory and recursively fetch subdirectories
        // Note: We return arrays instead of using inout because actors don't support inout parameters
        return try await fetchFilesRecursively(modelId: modelId, path: "")
    }

    private func fetchFilesRecursively(
        modelId: String,
        path: String
    ) async throws -> [HFFile] {
        var files: [HFFile] = []

        // HuggingFace API endpoint for tree
        let pathSuffix = path.isEmpty ? "" : "/\(path)"
        let apiURLString = "https://huggingface.co/api/models/\(modelId)/tree/main\(pathSuffix)"
        guard let apiURL = URL(string: apiURLString) else {
            throw DownloadError.invalidURL
        }

        logDebug("Fetching HF tree: \(apiURLString)", category: .download)

        let (data, response) = try await urlSession.data(from: apiURL)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            logWarning("Failed to fetch tree for path: \(path), status: \((response as? HTTPURLResponse)?.statusCode ?? 0)", category: .download)
            throw DownloadError.fileListFailed
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw DownloadError.fileListFailed
        }

        logDebug("Found \(json.count) items at path: \(path.isEmpty ? "/" : path)", category: .download)

        for item in json {
            guard let itemPath = item["path"] as? String,
                  let type = item["type"] as? String else { continue }

            if type == "directory" {
                // Check if this is a directory we need
                let dirName = (itemPath as NSString).lastPathComponent

                // Always recurse into:
                // - .mlmodelc directories (compiled CoreML models)
                // - Directories inside .mlmodelc
                // - "weights" directories
                // - Root-level directories (to find nested models)
                let shouldRecurse = dirName.hasSuffix(".mlmodelc") ||
                                   itemPath.contains(".mlmodelc") ||
                                   dirName == "weights" ||
                                   path.isEmpty  // Recurse into all root-level dirs

                if shouldRecurse {
                    logDebug("Recursing into directory: \(itemPath)", category: .download)
                    // Recursively fetch contents of this directory
                    let subFiles = try await fetchFilesRecursively(
                        modelId: modelId,
                        path: itemPath
                    )
                    files.append(contentsOf: subFiles)
                }
            } else if type == "file" {
                guard let size = item["size"] as? Int64 else { continue }

                // Check if this is an essential file
                let fileName = (itemPath as NSString).lastPathComponent
                let ext = (fileName as NSString).pathExtension.lowercased()

                let isEssential =
                    // Files inside mlmodelc directories
                    itemPath.contains(".mlmodelc/") ||
                    // Root essential files
                    fileName == "tokenizer.json" ||
                    fileName == "config.json" ||
                    fileName == "meta.yaml" ||
                    fileName == "tokenizer_config.json" ||
                    fileName == "tokenizer.model" ||
                    fileName == "vocab.json" ||
                    fileName == "merges.txt" ||
                    fileName == "special_tokens_map.json" ||
                    // By extension
                    ext == "yaml" ||
                    ext == "bin"

                if isEssential {
                    let encodedPath = itemPath.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? itemPath
                    let downloadURL = URL(string: "https://huggingface.co/\(modelId)/resolve/main/\(encodedPath)")!
                    files.append(HFFile(name: itemPath, url: downloadURL, size: size))
                    logDebug("Added file: \(itemPath) (\(ByteCountFormatter.string(fromByteCount: size, countStyle: .file)))", category: .download)
                }
            }
        }

        return files
    }

    // MARK: - File Download

    private func downloadFile(_ file: HFFile, to directory: URL, modelId: String) async throws {
        let destURL = directory.appendingPathComponent(file.name)

        // Create subdirectories if needed
        let destDir = destURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        // Check if file exists and is complete
        if FileManager.default.fileExists(atPath: destURL.path) {
            if let attrs = try? FileManager.default.attributesOfItem(atPath: destURL.path),
               let existingSize = attrs[.size] as? Int64,
               existingSize == file.size {
                logDebug("Skipping already downloaded: \(file.name)", category: .download)
                downloadedBytesTotal[modelId] = (downloadedBytesTotal[modelId] ?? 0) + file.size
                updateProgress(for: modelId, bytesWritten: 0)
                return
            }
        }

        logDebug("Downloading: \(file.name) (\(ByteCountFormatter.string(fromByteCount: file.size, countStyle: .file)))", category: .download)

        // Create the download task with completion handler
        let task = urlSession.downloadTask(with: file.url) { [weak self] tempURL, response, error in
            Task {
                if let error = error {
                    await self?.handleDownloadCompletion(modelId: modelId, file: file, destURL: destURL, result: .failure(error))
                    return
                }

                guard let tempURL = tempURL else {
                    await self?.handleDownloadCompletion(modelId: modelId, file: file, destURL: destURL, result: .failure(DownloadError.downloadFailed(file.name)))
                    return
                }

                do {
                    // Remove existing file if any
                    try? FileManager.default.removeItem(at: destURL)
                    // Move downloaded file to destination
                    try FileManager.default.moveItem(at: tempURL, to: destURL)
                    await self?.handleDownloadCompletion(modelId: modelId, file: file, destURL: destURL, result: .success(()))
                } catch {
                    await self?.handleDownloadCompletion(modelId: modelId, file: file, destURL: destURL, result: .failure(error))
                }
            }
        }

        // Set up progress tracking BEFORE starting the download (critical fix!)
        downloadTasks[modelId] = task
        observeProgress(task: task, modelId: modelId)

        return try await withCheckedThrowingContinuation { continuation in
            downloadCompletions[modelId] = continuation
            task.resume()
        }
    }

    private func handleDownloadCompletion(modelId: String, file: HFFile, destURL: URL, result: Result<Void, Error>) {
        switch result {
        case .success:
            addDownloadedBytes(file.size, for: modelId)
            downloadCompletions[modelId]?.resume()
        case .failure(let error):
            if let dlError = error as? DownloadError {
                downloadCompletions[modelId]?.resume(throwing: dlError)
            } else {
                downloadCompletions[modelId]?.resume(throwing: DownloadError.networkError(error))
            }
        }
        downloadCompletions.removeValue(forKey: modelId)
    }

    // Timer for progress polling (needed for macOS where KVO may not fire frequently)
    private var progressTimers: [String: Task<Void, Never>] = [:]

    private func observeProgress(task: URLSessionDownloadTask, modelId: String) {
        // Use KVO to observe countOfBytesReceived directly on the task (more reliable than progress object)
        let observation = task.observe(\.countOfBytesReceived) { [weak self] task, _ in
            Task {
                await self?.handleProgressUpdate(
                    for: modelId,
                    completedBytes: task.countOfBytesReceived,
                    totalBytes: task.countOfBytesExpectedToReceive
                )
            }
        }

        // Store observation to keep it alive for the duration of the download
        progressObservations[modelId] = observation

        // Start a timer-based polling as backup (KVO may not fire frequently on iOS or macOS)
        progressTimers[modelId]?.cancel()
        progressTimers[modelId] = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 300_000_000) // 0.3 second for more frequent updates
                guard !Task.isCancelled else { break }
                let completed = task.countOfBytesReceived
                let total = task.countOfBytesExpectedToReceive
                if completed > 0 {
                    await self?.handleProgressUpdate(for: modelId, completedBytes: completed, totalBytes: total)
                }
            }
        }
    }

    private func handleProgressUpdate(for modelId: String, completedBytes: Int64, totalBytes: Int64) {
        let now = Date()

        // Throttle: only update at most every 300ms to avoid excessive UI refreshes
        if let lastUpdate = lastProgressUpdate[modelId],
           now.timeIntervalSince(lastUpdate) < 0.3 {
            return
        }
        lastProgressUpdate[modelId] = now

        // Update current file progress and notify
        let baseDownloaded = downloadedBytesTotal[modelId] ?? 0
        let currentTotal = baseDownloaded + completedBytes
        let total = totalBytesExpected[modelId] ?? 0

        // Calculate speed using rolling average
        var history = downloadedBytesHistory[modelId] ?? []
        history.append((now, currentTotal))

        // Keep last 5 seconds of history for more responsive updates
        history = history.filter { now.timeIntervalSince($0.0) < 5 }
        downloadedBytesHistory[modelId] = history

        var speed: Double = 0
        if history.count >= 2 {
            // Use oldest and newest for better average
            let oldest = history.first!
            let newest = history.last!
            let timeDiff = newest.0.timeIntervalSince(oldest.0)
            let bytesDiff = newest.1 - oldest.1
            if timeDiff > 0.1 {  // Need at least 100ms of data
                speed = Double(bytesDiff) / timeDiff
            }
        }

        // Fallback: calculate from start time if no recent speed
        if speed == 0, let startTime = downloadStartTime[modelId], currentTotal > 0 {
            let totalTime = now.timeIntervalSince(startTime)
            if totalTime > 0 {
                speed = Double(currentTotal) / totalTime
            }
        }

        let progress = DownloadProgress(
            totalBytes: total,
            downloadedBytes: currentTotal,
            currentFile: currentFileName[modelId] ?? "",
            filesCompleted: currentFileIndex[modelId] ?? 0,
            totalFiles: totalFilesCount[modelId] ?? 0,
            bytesPerSecond: speed
        )

        progressCallbacks[modelId]?(progress)
    }

    private func addDownloadedBytes(_ bytes: Int64, for modelId: String) {
        downloadedBytesTotal[modelId] = (downloadedBytesTotal[modelId] ?? 0) + bytes
    }

    private func updateProgress(for modelId: String, bytesWritten: Int64) {
        let now = Date()
        let total = totalBytesExpected[modelId] ?? 0
        let downloaded = downloadedBytesTotal[modelId] ?? 0

        // Calculate speed
        var history = downloadedBytesHistory[modelId] ?? []
        history.append((now, downloaded))

        // Keep last 10 seconds of history
        history = history.filter { now.timeIntervalSince($0.0) < 10 }
        downloadedBytesHistory[modelId] = history

        let speed: Double
        if history.count >= 2,
           let first = history.first,
           let last = history.last {
            let timeDiff = last.0.timeIntervalSince(first.0)
            let bytesDiff = last.1 - first.1
            speed = timeDiff > 0 ? Double(bytesDiff) / timeDiff : 0
        } else {
            speed = 0
        }

        let progress = DownloadProgress(
            totalBytes: total,
            downloadedBytes: downloaded,
            currentFile: currentFileName[modelId] ?? "",
            filesCompleted: currentFileIndex[modelId] ?? 0,
            totalFiles: totalFilesCount[modelId] ?? 0,
            bytesPerSecond: speed
        )

        progressCallbacks[modelId]?(progress)
    }

    // MARK: - Verification

    private func verifyDownload(modelId: String, at directory: URL) async -> Bool {
        // Check for meta.yaml (required)
        let metaPath = directory.appendingPathComponent("meta.yaml")
        if !FileManager.default.fileExists(atPath: metaPath.path) {
            logWarning("Missing required file: meta.yaml", category: .download)
            return false
        }

        // Check for tokenizer (either .json or .model is acceptable)
        let tokenizerJson = directory.appendingPathComponent("tokenizer.json")
        let tokenizerModel = directory.appendingPathComponent("tokenizer.model")
        if !FileManager.default.fileExists(atPath: tokenizerJson.path) &&
           !FileManager.default.fileExists(atPath: tokenizerModel.path) {
            logWarning("Missing tokenizer file (neither tokenizer.json nor tokenizer.model found)", category: .download)
            return false
        }

        // Check for .mlmodelc directories
        let contents = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let hasModel = contents?.contains(where: { $0.pathExtension == "mlmodelc" }) ?? false

        if !hasModel {
            logWarning("No .mlmodelc model found", category: .download)
            return false
        }

        return true
    }

    // MARK: - Helpers

    private func storeResumeData(_ data: Data?, for modelId: String) {
        if let data = data {
            resumeData[modelId] = data
        }
    }

    private func cleanup(modelId: String) {
        downloadTasks.removeValue(forKey: modelId)
        progressCallbacks.removeValue(forKey: modelId)
        completionCallbacks.removeValue(forKey: modelId)
        downloadStartTime.removeValue(forKey: modelId)
        downloadedBytesHistory.removeValue(forKey: modelId)
        currentFileIndex.removeValue(forKey: modelId)
        totalFilesCount.removeValue(forKey: modelId)
        totalBytesExpected.removeValue(forKey: modelId)
        downloadedBytesTotal.removeValue(forKey: modelId)
        currentFileName.removeValue(forKey: modelId)
        progressObservations.removeValue(forKey: modelId)
        downloadCompletions.removeValue(forKey: modelId)
        currentFileSize.removeValue(forKey: modelId)
        lastProgressUpdate.removeValue(forKey: modelId)

        // Cancel progress timer (macOS)
        progressTimers[modelId]?.cancel()
        progressTimers.removeValue(forKey: modelId)
    }
}
