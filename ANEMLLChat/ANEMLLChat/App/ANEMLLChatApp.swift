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
                // Force dark mode to match hardcoded dark backgrounds throughout the app
                .preferredColorScheme(.dark)
        }
        #if os(macOS)
        // Use titleBar style to show toolbar
        .windowStyle(.titleBar)
        .defaultSize(width: 1000, height: 700)
        .commands {
            CommandGroup(replacing: .appInfo) {
                AboutCommands()
            }
        }
        #endif

        #if os(macOS)
        Settings {
            SettingsView()
                .environment(chatViewModel)
                .environment(modelManager)
        }

        Window("Acknowledgements", id: "acknowledgements") {
            AcknowledgementsView()
        }
        .defaultSize(width: 450, height: 400)
        .windowResizability(.contentSize)
        #endif
    }
}

#if os(macOS)
struct AboutCommands: View {
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button("About ANEMLL Chat") {
            NSApplication.shared.orderFrontStandardAboutPanel(options: [
                .credits: NSAttributedString(
                    string: "On-device LLM inference powered by Apple Neural Engine",
                    attributes: [
                        .font: NSFont.systemFont(ofSize: 11),
                        .foregroundColor: NSColor.secondaryLabelColor
                    ]
                )
            ])
        }
        Button("Acknowledgements...") {
            openWindow(id: "acknowledgements")
        }
    }
}
#endif
