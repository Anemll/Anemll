//
//  InputBar.swift
//  ANEMLLChat
//
//  Text input with send button
//

import SwiftUI

struct InputBar: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(ModelManagerViewModel.self) private var modelManager

    @FocusState private var isFocused: Bool
    @State private var showLoadingToast = false

    var body: some View {
        @Bindable var vm = chatVM

        ZStack(alignment: .top) {
            HStack(alignment: .bottom, spacing: 12) {
                // Text field
                textField

                // Send/Stop button
                sendButton
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 26, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 26, style: .continuous)
                    .stroke(inputBarBorder, lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.2), radius: 12, y: 6)

            // Toast overlay - appears above input bar
            if showLoadingToast {
                LoadingToastView(message: "Model still loading...")
                    .transition(.asymmetric(
                        insertion: .move(edge: .top).combined(with: .opacity),
                        removal: .opacity
                    ))
                    .offset(y: -50)
            }
        }
    }

    // MARK: - Text Field

    private var textField: some View {
        @Bindable var vm = chatVM

        return TextField("Message...", text: $vm.inputText, axis: .vertical)
            .textFieldStyle(.plain)
            .lineLimit(1...6)
            .focused($isFocused)
            .disabled(chatVM.isGenerating)
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(inputFieldBackground)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .stroke(inputFieldBorder, lineWidth: 1)
            )
            .onSubmit {
                sendMessage()
            }
            #if os(macOS)
            .onKeyPress(.return, phases: .down) { _ in
                if !NSEvent.modifierFlags.contains(.shift) {
                    sendMessage()
                    return .handled
                }
                return .ignored
            }
            #endif
    }

    // MARK: - Send Button

    private var sendButton: some View {
        Button {
            if chatVM.isGenerating {
                chatVM.cancelGeneration()
            } else {
                sendMessage()
            }
        } label: {
            Image(systemName: chatVM.isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                .font(.title)
                .foregroundStyle(buttonColor)
        }
        .buttonStyle(.plain)
        .disabled(!canSend && !chatVM.isGenerating)
        .keyboardShortcut(.return, modifiers: .command)
        .help(chatVM.isGenerating ? "Stop generation" : "Send message (⌘ Return)")
    }

    private var canSend: Bool {
        !chatVM.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !chatVM.isGenerating &&
        !modelManager.isLoadingModel
    }

    private var buttonColor: Color {
        if chatVM.isGenerating {
            return .red
        }
        return canSend ? .accentColor : .secondary.opacity(0.5)
    }

    // MARK: - Actions

    private func sendMessage() {
        // Check if model is still loading
        if modelManager.isLoadingModel {
            showToast()
            return
        }

        guard canSend else { return }

        Task {
            await chatVM.sendMessage()
        }

        isFocused = false
    }

    private func showToast() {
        withAnimation(.easeOut(duration: 0.2)) {
            showLoadingToast = true
        }

        // Auto-dismiss after 2 seconds
        Task {
            try? await Task.sleep(for: .seconds(2))
            withAnimation(.easeIn(duration: 0.3)) {
                showLoadingToast = false
            }
        }
    }
}

// MARK: - Platform Colors

#if os(iOS)
private let inputFieldBackground = Color.white.opacity(0.08)
private let inputFieldBorder = Color.white.opacity(0.12)
private let inputBarBorder = Color.white.opacity(0.12)
#else
private let platformTertiaryBackground = NSColor.textBackgroundColor
private let inputFieldBackground = Color(platformTertiaryBackground)
private let inputFieldBorder = Color.secondary.opacity(0.3)
private let inputBarBorder = Color.secondary.opacity(0.2)
#endif

// MARK: - Loading Toast View

private struct LoadingToastView: View {
    let message: String

    var body: some View {
        HStack(spacing: 8) {
            ProgressView()
                .controlSize(.small)

            Text(message)
                .font(.subheadline)
                .fontWeight(.medium)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: Capsule())
        .shadow(color: .black.opacity(0.1), radius: 4, y: 2)
    }
}

#Preview {
    VStack {
        Spacer()
        InputBar()
            .environment(ChatViewModel())
            .environment(ModelManagerViewModel())
    }
}
