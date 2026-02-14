// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ANEMLLChat",
    platforms: [
        .iOS(.v18),
        .macOS(.v15)
    ],
    products: [
        .executable(
            name: "ANEMLLChat",
            targets: ["ANEMLLChat"]
        )
    ],
    dependencies: [
        .package(path: "../anemll-swift-cli"),
        .package(url: "https://github.com/jpsim/Yams.git", from: "5.0.0")
    ],
    targets: [
        .executableTarget(
            name: "ANEMLLChat",
            dependencies: [
                .product(name: "AnemllCore", package: "anemll-swift-cli"),
                "Yams"
            ],
            path: "ANEMLLChat",
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]
        )
    ]
)
