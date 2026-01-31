#!/usr/bin/env swift

import Foundation

// Simple test of HuggingFace download logic

struct HFFile {
    let name: String
    let url: URL
    let size: Int64
}

func fetchFilesRecursively(modelId: String, path: String = "") async throws -> [HFFile] {
    var files: [HFFile] = []

    let pathSuffix = path.isEmpty ? "" : "/\(path)"
    let apiURLString = "https://huggingface.co/api/models/\(modelId)/tree/main\(pathSuffix)"

    guard let apiURL = URL(string: apiURLString) else {
        print("Invalid URL: \(apiURLString)")
        return []
    }

    print("Fetching: \(apiURLString)")

    let (data, response) = try await URLSession.shared.data(from: apiURL)

    guard let httpResponse = response as? HTTPURLResponse else {
        print("Not HTTP response")
        return []
    }

    print("  Status: \(httpResponse.statusCode)")

    guard httpResponse.statusCode == 200 else {
        print("  Failed with status: \(httpResponse.statusCode)")
        return []
    }

    guard let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
        print("  Failed to parse JSON")
        return []
    }

    print("  Found \(json.count) items")

    for item in json {
        guard let itemPath = item["path"] as? String,
              let type = item["type"] as? String else { continue }

        if type == "directory" {
            let dirName = (itemPath as NSString).lastPathComponent
            let shouldRecurse = dirName.hasSuffix(".mlmodelc") ||
                               itemPath.contains(".mlmodelc") ||
                               dirName == "weights"

            if shouldRecurse {
                print("  -> Recursing into: \(itemPath)")
                let subFiles = try await fetchFilesRecursively(modelId: modelId, path: itemPath)
                files.append(contentsOf: subFiles)
            }
        } else if type == "file" {
            guard let size = item["size"] as? Int64 else { continue }

            let fileName = (itemPath as NSString).lastPathComponent
            let ext = (fileName as NSString).pathExtension.lowercased()

            let isEssential =
                itemPath.contains(".mlmodelc/") ||
                fileName == "tokenizer.json" ||
                fileName == "config.json" ||
                fileName == "meta.yaml" ||
                fileName == "tokenizer_config.json" ||
                fileName == "tokenizer.model" ||
                fileName == "vocab.json" ||
                fileName == "merges.txt" ||
                fileName == "special_tokens_map.json" ||
                ext == "yaml" ||
                ext == "bin"

            if isEssential {
                let encodedPath = itemPath.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? itemPath
                let downloadURL = URL(string: "https://huggingface.co/\(modelId)/resolve/main/\(encodedPath)")!
                files.append(HFFile(name: itemPath, url: downloadURL, size: size))
                print("  + File: \(itemPath) (\(ByteCountFormatter.string(fromByteCount: size, countStyle: .file)))")
            }
        }
    }

    return files
}

// Run the test
let modelId = "anemll/anemll-llama-3.2-1B-iOSv2.0"

print("Testing download for: \(modelId)\n")

let semaphore = DispatchSemaphore(value: 0)

Task {
    do {
        let files = try await fetchFilesRecursively(modelId: modelId)

        print("\n=== SUMMARY ===")
        print("Total files: \(files.count)")

        let totalSize = files.reduce(0) { $0 + $1.size }
        print("Total size: \(ByteCountFormatter.string(fromByteCount: totalSize, countStyle: .file))")

        print("\nFiles to download:")
        for file in files {
            print("  \(file.name)")
        }

    } catch {
        print("Error: \(error)")
    }
    semaphore.signal()
}

semaphore.wait()
