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

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(chatViewModel)
                .environment(modelManager)
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
