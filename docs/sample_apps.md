# iOS/macOS Sample Applications

## Overview
The sample applications demonstrate how to integrate ANEMLL models into iOS and macOS applications. These apps provide a chat interface similar to popular LLM applications but running completely on-device using the Apple Neural Engine.

## Project Location
```bash
./anemll-chatbot/
```

## Target Audience
- iOS/macOS developers looking to integrate on-device LLMs
- Developers wanting to understand ANE model integration
- Anyone interested in building privacy-focused AI applications

## Features
- SwiftUI-based chat interface
- Supports both iOS and macOS platforms
- On-device inference using Apple Neural Engine
- Conversation history management
- Model download and management
- Progress indicators for long responses
- Token usage statistics

## Requirements
- Xcode 15.0 or later
- iOS 18.0 / macOS 15.0 or later
- For macOS: M1 chip or newer
- For iOS: A14 Bionic or newer (iPhone 12 and later)
- Minimum 4GB of free storage space for models

> [!Note]
> Older 8-core iOS ANE devices may experience compatibility issues.

## Model Compatibility
- 1B models: Compatible with most supported devices
- 3B models: Requires device with 8GB RAM
- 8B DeepSeek models: Only compatible with iPad Pro 16GB variants

## Configuration

### 1. Project Setup
```bash
cd ./anemll-chatbot/
open anemll-chatbot.xcodeproj 
```

> [!Important]
> Remember to change the Xcode Team and Bundle ID if you're building the project yourself.
> Selecf iOS or My Mac target device and compile
> for macOS we use MacCatalyst build target
> target : anemll-chatbot

### 2. Model Management
The app supports several ways to use models:

1. **Default Model**: App automatically downloads a 1B parameter model on first launch
2. **Custom HuggingFace Models**: Add custom models via HuggingFace URL
3. **Local Models**: Copy MLModels directly to your device using the Files app (iOS)

To switch between models:
1. Select the desired model from the model list
2. Click "Load Model" to load it into memory

### 3. Building the Project

#### For iOS:
1. Select your target iOS device or simulator
2. Choose the "anemll-chatbot" scheme
3. Build and run (⌘R)

#### For macOS:
1. Select "My Mac" as the target
2. Choose the "anemll-chatbot" scheme and My Mac (MacCatalys)
3. Build and run (⌘R)

## Testing

### Using TestFlight
1. Download our TestFlight app for quick testing:
   [TestFlight Link](https://testflight.apple.com/join/jrQq1D1C)
2. The app comes with a default 1B parameter model
3. Additional models can be downloaded from our Hugging Face repository
4. Custom models can be added via HuggingFace URLs

> [!Note]
> This is an early beta release. We are actively working on:
> - Improving quantization for reference models
> - Fixing known issues
> - Enhancing model conversion scripts

## Custom Model Integration
You can use your own models in two ways:
1. Upload unzipped MLModels to HuggingFace then add Custom Model in the App useing repo's URL
2. Copy models directly to your device using the Files app (iOS)
3. Follow the instructions in [prepare_hf.md](prepare_hf.md) to prepare your model for inference. For iOS deployment, make sure to use the `--ios` flag when converting the model.


## Customization
- Modify `ChatView.swift` to customize the chat interface
- Adjust model parameters in `InferenceEngine.swift`
- Configure model download options in `ModelManager.swift`

## Known Limitations
- iOS devices are limited to 1GB per chunk, so we need to split model in chunks
- Initial model loading may take a few seconds

## Support
For issues and questions:
- Create an issue on GitHub
- Contact us at [realanemll@gmail.com](mailto:realanemll@gmail.com)
- Follow updates on [@anemll](https://x.com/anemll)
