# ANEMLL Chat

A modern, lightweight SwiftUI chat application for on-device LLM inference using Apple Neural Engine.

## Features

- **On-Device Inference**: Run LLMs locally using CoreML and Apple Neural Engine
- **Model Management**: Download and manage models from HuggingFace
- **Streaming Responses**: Real-time token streaming with performance metrics
- **Multi-Platform**: iOS 18+ and macOS 15+ support
- **Modern SwiftUI**: Built with Swift 6 concurrency and Observation framework

## Architecture

```
ANEMLLChat/
├── App/
│   ├── ANEMLLChatApp.swift      # App entry point
│   └── ContentView.swift        # Root navigation
├── Models/
│   ├── ChatMessage.swift        # Message model
│   ├── Conversation.swift       # Conversation container
│   └── ModelInfo.swift          # Model metadata
├── Services/
│   ├── InferenceService.swift   # AnemllCore wrapper
│   ├── DownloadService.swift    # HuggingFace downloads
│   ├── StorageService.swift     # Persistence
│   └── Logger.swift             # Centralized logging
├── ViewModels/
│   ├── ChatViewModel.swift      # Chat state management
│   └── ModelManagerViewModel.swift
└── Views/
    ├── Chat/                    # Chat interface
    ├── Models/                  # Model management
    └── Settings/                # App settings
```

## Requirements

- **iOS**: 18.0+
- **macOS**: 15.0+
- **Xcode**: 16.0+
- **Swift**: 6.0+

## Building

### Using Swift Package Manager

```bash
cd ANEMLLChat
swift build
```

### Using Xcode

1. Open `ANEMLLChat` folder in Xcode
2. Select your target device
3. Build and run (⌘R)

## Dependencies

- **AnemllCore**: Local package from `anemll-swift-cli`
- **Yams**: YAML parsing for model configuration

## Usage

1. **Download a Model**: Open the Models panel and download a model from HuggingFace
2. **Load Model**: Tap on a downloaded model to load it
3. **Start Chatting**: Create a new conversation and send messages

## Default Models

- LLaMA 3.2 1B (optimized for iOS)
- DeepHermes 3B
- Qwen 3 0.6B

## Configuration

### Generation Settings

- **Temperature**: Control randomness (0.0-2.0)
- **Max Tokens**: Maximum response length (64-2048)
- **System Prompt**: Initial instructions for the assistant

### Adding Custom Models

1. Open Models panel
2. Tap "+" to add a custom model
3. Enter HuggingFace repo ID (e.g., `anemll/my-custom-model`)

### Local Models (Mac → iOS, no Hugging Face)

If your CoreML/ANE models live on your Mac, you can avoid Hugging Face entirely by either
copying them directly to the iOS app sandbox or serving them over your local network.

#### Option A: macOS local import (drag & drop) + AirDrop to iOS

If you want a “local model” workflow similar to the macOS app:

1. **macOS app**: allow dropping a compiled model folder into the app.
   - The app should prompt to **Import** (copy into app storage) or **Link** (reference the
     original folder for quick local testing).
2. **macOS app**: add a **Share** action in the Models view to AirDrop the model folder
   (or a zipped package) to iOS.
3. **iOS app**: handle incoming AirDrop files by unzipping (if needed) and moving the
   model into:
   - `Documents/Models/<model-id>/`
   - Update `Documents/models.json` so it appears in the list.

This flow is useful even without network transfer and matches a “local first” workflow.

#### Option B: Copy directly to iOS (USB / Files app)

1. Convert + compile your model for iOS.
2. Copy the entire model folder into the app's Documents directory:
   - iOS path: `Documents/Models/<model-id>/`
3. Ensure the folder contains `meta.yaml`, tokenizer files, and compiled `.mlmodelc` directories.
4. Add the model to the registry:
   - iOS path: `Documents/models.json`

This mirrors the existing on-device model discovery and avoids any network transfer.

#### Option C: Serve from macOS and pull over Wi-Fi (VHTTP-style, requires app support)

1. Put the compiled model directory in a single folder on your Mac.
2. Start a simple HTTP server in that folder (example):

   ```bash
   cd /path/to/Models
   python3 -m http.server 8080
   ```

3. Note your Mac’s IP address (e.g., `192.168.1.50`).
4. **Important:** the current iOS “Add Model” flow only accepts a Hugging Face repo ID.
   To use a LAN-served model, you’ll need to extend the app to accept a base URL.
   Suggested code touchpoints:
   - `AddModelView`: allow URL input alongside HF repo IDs.
   - `ModelManagerViewModel.addCustomModel`: store a `baseURL` or `sourceType`.
   - `DownloadService`: resolve file URLs from either Hugging Face or your local server.
5. Once that support exists, the iOS app can download directly from:
   - Example: `http://192.168.1.50:8080/<model-id>/`

If you want zero code changes today, use **Option A** (copy directly into
`Documents/Models/<model-id>/` and update `Documents/models.json`).

## Performance Metrics

The app displays real-time metrics during generation:

- **Tokens/sec**: Generation speed
- **Token count**: Total tokens generated
- **Window shifts**: Context window rotations (for long conversations)

## Release Notes

- Fixed Qwen3 multi-chunk inference divergence caused by applying final normalization on every FFN chunk. Final normalization is now applied only on the last chunk, restoring stable Qwen3 chunked generation quality.

## License

MIT License - See LICENSE file for details

## Credits

- [ANEMLL](https://github.com/anemll/anemll) - Apple Neural Engine LLM framework
- [AnemllCore](../anemll-swift-cli) - Swift inference library
