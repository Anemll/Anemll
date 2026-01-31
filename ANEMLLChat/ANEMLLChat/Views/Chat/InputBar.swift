//
//  InputBar.swift
//  ANEMLLChat
//
//  Text input with send button
//

import SwiftUI

struct InputBar: View {
    @Environment(ChatViewModel.self) private var chatVM

    @FocusState private var isFocused: Bool

    var body: some View {
        @Bindable var vm = chatVM

        HStack(alignment: .bottom, spacing: 12) {
            // Text field
            textField

            // Send/Stop button
            sendButton
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color(platformSecondaryBackground))
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
                    .fill(Color(platformTertiaryBackground))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
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
        !chatVM.isGenerating
    }

    private var buttonColor: Color {
        if chatVM.isGenerating {
            return .red
        }
        return canSend ? .accentColor : .secondary.opacity(0.5)
    }

    // MARK: - Actions

    private func sendMessage() {
        guard canSend else { return }

        Task {
            await chatVM.sendMessage()
        }

        isFocused = false
    }
}

// MARK: - Platform Colors

#if os(iOS)
private let platformSecondaryBackground = UIColor.secondarySystemBackground
private let platformTertiaryBackground = UIColor.tertiarySystemBackground
#else
private let platformSecondaryBackground = NSColor.controlBackgroundColor
private let platformTertiaryBackground = NSColor.textBackgroundColor
#endif

#Preview {
    VStack {
        Spacer()
        InputBar()
            .environment(ChatViewModel())
    }
}
