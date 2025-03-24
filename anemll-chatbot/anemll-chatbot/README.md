# Anemll Chatbot

A conversational AI chatbot for iOS powered by the Anemll CoreML models.

## Development

### Project Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for details on the project architecture and development guidelines. 

**Important architecture rules:**
- Follow the DRY principle by reusing functionality from AnemllCore rather than duplicating it
- Use YAMLConfig for all path generation and model configuration
- Maintain clear separation between configuration parsing and file validation

### Dependencies

- **AnemllCore**: Core ML model loading, inference, and tokenization functionality
- **SwiftUI**: UI framework
- **Yams**: YAML parsing

### Building and Running

1. Clone the repository
2. Open the project in Xcode
3. Install dependencies
4. Build and run on an iOS device or simulator

## Features

- Chat with ML models running locally on device
- Support for multiple models
- Model management (download, delete)
- Settings customization 