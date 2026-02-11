//
//  ANEMLLChatApp.swift
//  ANEMLLChat
//
//  Modern SwiftUI app for ANEMLL CoreML inference
//

import SwiftUI

@main
struct ANEMLLChatApp: App {
    @State private var chatViewModel = ChatViewModel()
    @State private var modelManager = ModelManagerViewModel()

    init() {
        // Log device info at startup
        logInfo("=== ANEMLL Chat Starting ===", category: .app)
        logInfo("Device: \(DeviceType.deviceSummary)", category: .app)
        logInfo("App version: \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?") (\(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"))", category: .app)

        // Start UI freeze watchdog in debug builds
        #if DEBUG
        UIFreezeWatchdog.shared.start()
        #endif
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(chatViewModel)
                .environment(modelManager)
                .onOpenURL { url in
                    Task {
                        await modelManager.handleIncomingTransferURL(url)
                    }
                }
                #if os(iOS)
                // Force dark mode on iOS/iPadOS/visionOS to match hardcoded dark backgrounds
                .preferredColorScheme(.dark)
                #endif
        }
        #if os(macOS)
        // Use titleBar style to show toolbar
        .windowStyle(.titleBar)
        .defaultSize(width: 1000, height: 700)
        #endif

        #if os(macOS)
        Settings {
            SettingsView()
                .environment(chatViewModel)
                .environment(modelManager)
        }
        #endif
    }
}
