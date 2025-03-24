# Anemll Chatbot Project Rules

@files: **/*.swift

This file serves as an index for all project-specific rules and coding standards. These rules should be followed when developing the Anemll Chatbot application.

## Core Architecture Rules

@file: ../../ARCHITECTURE.md

See the full [ARCHITECTURE.md](../../ARCHITECTURE.md) document for detailed guidelines.

### Key Rules

1. **[Path Generation](./path-generation.md)**: Use AnemllCore's YAMLConfig for all path generation
2. **[Error Handling](./error-handling.md)**: Use custom error types with detailed messages
3. **Separation of Concerns**: Keep model configuration separate from validation
4. **Logging**: Follow consistent logging patterns

## File Organization

- `/Services`: Business logic, model interaction, and core functionality
- `/Views`: SwiftUI views and UI components
- `/Models`: Swift data models and types
- `/Utilities`: Helper functions and extensions

## Naming Conventions

- **Services**: Suffix with `Service` (e.g., `InferenceService`)
- **View Models**: Suffix with `ViewModel` (e.g., `ChatViewModel`)
- **SwiftUI Views**: Describe the view's purpose (e.g., `ModelManagementView`)
- **Extensions**: Name with `<Type>+<Functionality>` (e.g., `String+Utils`)

## Swift Guidelines

- Use Swift's async/await for asynchronous code when possible
- Avoid force unwrapping optionals (`!`) in production code
- Prefer value types (structs) over reference types (classes) where appropriate
- Use dependency injection to make code testable

## Commit Guidelines

- Use descriptive commit messages that explain the change
- Reference issue numbers in commits when applicable
- Keep commits focused on a single logical change 