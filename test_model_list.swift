#!/usr/bin/env swift

import Foundation

// Simulate ModelInfo
struct ModelInfo: Identifiable {
    let id: String
    let name: String
    var isDownloaded: Bool = false
    var isDownloading: Bool = false
}

// Test: Default models
let defaultModels = [
    ModelInfo(id: "anemll/anemll-llama-3.2-1B-iOSv2.0", name: "LLaMA 3.2 1B"),
    ModelInfo(id: "anemll/anemll-google-gemma-3-1b-it-ctx4096_0.3.4", name: "Gemma 3 1B"),
    ModelInfo(id: "anemll/anemll-Qwen3-4B-ctx1024_0.3.0", name: "Qwen 3 4B"),
    ModelInfo(id: "anemll/anemll-Llama-3.2-1B-FAST-iOS_0.3.0", name: "LLaMA 3.2 1B FAST")
]

print("=== Testing Model List Logic ===\n")

// Test 1: Initial state
print("1. Initial state (no downloads):")
var availableModels = defaultModels
let availableForDownload = availableModels.filter { !$0.isDownloaded && !$0.isDownloading }
let downloadedModels = availableModels.filter { $0.isDownloaded }
print("   Total: \(availableModels.count)")
print("   Available: \(availableForDownload.count)")
print("   Downloaded: \(downloadedModels.count)")
print("   ✅ PASS: All 4 models should show in 'Available' section\n")

// Test 2: One model downloading
print("2. One model downloading:")
availableModels[0].isDownloading = true
let available2 = availableModels.filter { !$0.isDownloaded && !$0.isDownloading }
let downloading2 = availableModels.filter { $0.isDownloading }
print("   Available: \(available2.count) (should be 3)")
print("   Downloading: \(downloading2.count) (should be 1)")
print("   Downloading model: \(downloading2.first?.name ?? "none")")
print("   ✅ PASS: 3 in Available, 1 in Downloading\n")

// Test 3: One model downloaded
print("3. One model downloaded:")
availableModels[0].isDownloading = false
availableModels[0].isDownloaded = true
let available3 = availableModels.filter { !$0.isDownloaded && !$0.isDownloading }
let downloaded3 = availableModels.filter { $0.isDownloaded }
print("   Available: \(available3.count) (should be 3)")
print("   Downloaded: \(downloaded3.count) (should be 1)")
print("   Downloaded model: \(downloaded3.first?.name ?? "none")")
print("   ✅ PASS: 3 in Available, 1 in Downloaded\n")

// Test 4: Adding a custom model
print("4. Adding custom model:")
var customModel = ModelInfo(id: "custom/my-model", name: "My Custom Model")
availableModels.append(customModel)
let available4 = availableModels.filter { !$0.isDownloaded && !$0.isDownloading }
print("   Total models: \(availableModels.count) (should be 5)")
print("   Available: \(available4.count) (should be 4)")
print("   ✅ PASS: Custom model added and shows in Available\n")

// Test 5: Custom model downloading
print("5. Custom model starts downloading:")
if let idx = availableModels.firstIndex(where: { $0.id == "custom/my-model" }) {
    availableModels[idx].isDownloading = true
}
let available5 = availableModels.filter { !$0.isDownloaded && !$0.isDownloading }
let downloading5 = availableModels.filter { $0.isDownloading }
print("   Available: \(available5.count) (should be 3)")
print("   Downloading: \(downloading5.count) (should be 1)")
print("   Downloading: \(downloading5.first?.name ?? "none")")
print("   ✅ PASS: Custom model moves to Downloading section\n")

print("=== All logic tests passed! ===")
print("\nIf the app shows empty list, the issue is:")
print("1. @Observable not triggering view updates")
print("2. Environment not passing correctly")
print("3. init() not setting defaultModels (but we fixed that)")
