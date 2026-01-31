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

## Performance Metrics

The app displays real-time metrics during generation:

- **Tokens/sec**: Generation speed
- **Token count**: Total tokens generated
- **Window shifts**: Context window rotations (for long conversations)

## License

MIT License - See LICENSE file for details

## Credits

- [ANEMLL](https://github.com/anemll/anemll) - Apple Neural Engine LLM framework
- [AnemllCore](../anemll-swift-cli) - Swift inference library
