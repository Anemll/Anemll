// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "anemll-swift-cli",
    platforms: [
        .macOS(.v15),  // macOS 15 (Sonoma)
        .iOS(.v18),    // iOS 18
    ],
    products: [
        .library(
            name: "AnemllCore",
            targets: ["AnemllCore"]
        ),
        .executable(
            name: "anemllcli",
            targets: ["ANEMLLCLI"]
        )
    ],
    dependencies: [
        // CLI argument parsing
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        // Transformers with tokenizers
        .package(url: "https://github.com/huggingface/swift-transformers", branch: "main"),  // Use latest from main branch
        // YAML parser
        .package(url: "https://github.com/jpsim/Yams.git", from: "5.0.0"),
        // Templating (similar to Jinja)
        .package(url: "https://github.com/stencilproject/Stencil.git", from: "0.14.0")
    ],
    targets: [
        .target(
            name: "AnemllCore",
            dependencies: [
                "Yams",
                .product(name: "Transformers", package: "swift-transformers"),
                "Stencil"
            ]
        ),
        .executableTarget(
            name: "ANEMLLCLI",
            dependencies: [
                "AnemllCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Transformers", package: "swift-transformers")
            ],
            path: "Sources/ANEMLLCLI",
            resources: [
                .process("Resources/anemll.entitlements"),
                .process("Resources/RunDestinations.plist"),
                .process("Resources/anemll-swift-cli.xcscheme")
            ]
        )
    ]
) 
