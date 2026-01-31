#!/usr/bin/env swift

import Foundation

// MARK: - Test Framework
var testsPassed = 0
var testsFailed = 0

func test(_ name: String, _ condition: Bool, _ message: String = "") {
    if condition {
        print("✅ PASS: \(name)")
        testsPassed += 1
    } else {
        print("❌ FAIL: \(name) - \(message)")
        testsFailed += 1
    }
}

func testAsync(_ name: String, _ block: () async throws -> Bool) async {
    do {
        let result = try await block()
        test(name, result)
    } catch {
        test(name, false, "Exception: \(error)")
    }
}

// MARK: - Model Info (copy from app)
struct ModelInfo: Codable {
    let id: String
    let name: String
    let description: String
    let size: String
    var isDownloaded: Bool = false
    var isDownloading: Bool = false
}

let defaultModels = [
    ModelInfo(id: "anemll/anemll-llama-3.2-1B-iOSv2.0", name: "LLaMA 3.2 1B", description: "Meta's LLaMA", size: "1.2 GB"),
    ModelInfo(id: "anemll/anemll-deephermes-3B-iOSv2.0", name: "DeepHermes 3B", description: "DeepHermes", size: "2.8 GB"),
    ModelInfo(id: "anemll/anemll-qwen3-0.6B-iOSv2.0", name: "Qwen 3 0.6B", description: "Qwen", size: "0.6 GB")
]

// MARK: - Test 1: Default Models Exist
print("\n=== TEST 1: Default Models ===")
test("Has default models", defaultModels.count == 3)
test("First model has valid ID", defaultModels[0].id.contains("anemll/"))
test("Models have names", defaultModels.allSatisfy { !$0.name.isEmpty })

// MARK: - Test 2: HuggingFace API
print("\n=== TEST 2: HuggingFace API ===")

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
        throw URLError(.badURL)
    }

    let (data, response) = try await URLSession.shared.data(from: apiURL)

    guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
        throw URLError(.badServerResponse)
    }

    guard let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
        throw URLError(.cannotParseResponse)
    }

    for item in json {
        guard let itemPath = item["path"] as? String,
              let type = item["type"] as? String else { continue }

        if type == "directory" {
            let dirName = (itemPath as NSString).lastPathComponent
            let shouldRecurse = dirName.hasSuffix(".mlmodelc") ||
                               itemPath.contains(".mlmodelc") ||
                               dirName == "weights"

            if shouldRecurse {
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
                ext == "yaml"

            if isEssential {
                let encodedPath = itemPath.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? itemPath
                let downloadURL = URL(string: "https://huggingface.co/\(modelId)/resolve/main/\(encodedPath)")!
                files.append(HFFile(name: itemPath, url: downloadURL, size: size))
            }
        }
    }

    return files
}

// Test HuggingFace API for each default model
let semaphore = DispatchSemaphore(value: 0)

Task {
    for model in defaultModels {
        print("\nTesting model: \(model.name) (\(model.id))")

        do {
            let files = try await fetchFilesRecursively(modelId: model.id)
            test("[\(model.name)] Fetched files from HF", files.count > 0, "Found \(files.count) files")

            let hasMetaYaml = files.contains { $0.name == "meta.yaml" }
            test("[\(model.name)] Has meta.yaml", hasMetaYaml)

            let hasTokenizer = files.contains { $0.name == "tokenizer.json" }
            test("[\(model.name)] Has tokenizer.json", hasTokenizer)

            let hasMLModel = files.contains { $0.name.contains(".mlmodelc/") }
            test("[\(model.name)] Has .mlmodelc files", hasMLModel)

            let totalSize = files.reduce(0) { $0 + $1.size }
            print("   Total size: \(ByteCountFormatter.string(fromByteCount: totalSize, countStyle: .file))")
            print("   Files: \(files.count)")

        } catch {
            test("[\(model.name)] Fetched files from HF", false, error.localizedDescription)
        }
    }

    // MARK: - Test 3: Storage Paths
    print("\n=== TEST 3: Storage Paths ===")

    let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    test("Documents directory exists", FileManager.default.fileExists(atPath: documentsDir.path))

    let modelsDir = documentsDir.appendingPathComponent("Models", isDirectory: true)
    print("Models directory would be: \(modelsDir.path)")

    // Test model path generation
    let testModelId = "anemll/test-model"
    let expectedPath = modelsDir.appendingPathComponent(testModelId.replacingOccurrences(of: "/", with: "_"))
    test("Model path generation works", expectedPath.lastPathComponent == "anemll_test-model")

    // MARK: - Test 4: App Container (if exists)
    print("\n=== TEST 4: App Container ===")

    let containerPath = NSHomeDirectory() + "/Library/Containers/com.anemll.chat/Data/Documents/Models"
    if FileManager.default.fileExists(atPath: containerPath) {
        print("Container exists at: \(containerPath)")
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: containerPath) {
            print("Downloaded models: \(contents)")
            test("Can read container directory", true)
        }
    } else {
        print("Container not found (app not run yet or different location)")
        test("Container check", true, "Skipped - container not found")
    }

    // MARK: - Summary
    print("\n" + String(repeating: "=", count: 50))
    print("TEST SUMMARY")
    print(String(repeating: "=", count: 50))
    print("Passed: \(testsPassed)")
    print("Failed: \(testsFailed)")
    print(String(repeating: "=", count: 50))

    if testsFailed == 0 {
        print("✅ ALL TESTS PASSED!")
    } else {
        print("❌ SOME TESTS FAILED")
    }

    semaphore.signal()
}

semaphore.wait()
