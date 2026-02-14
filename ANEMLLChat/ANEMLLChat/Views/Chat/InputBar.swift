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
    @State private var showNoModelToast = false
    @State private var showMicErrorToast = false
    @State private var micErrorText = ""
    @State private var speechService = SpeechRecognitionService.shared
    @State private var textBeforeSpeech = ""  // Text in input before speech started
    @State private var isSpeechActive = false  // Track speech session to ignore recognizedText clearing
    @AppStorage("sendButtonOnLeft") private var sendButtonOnLeft = false
    @AppStorage("largeControls") private var largeControls = StorageService.defaultLargeControlsValue
    @AppStorage("showMicrophone") private var showMicrophone = true

    private var actionButtonSize: CGFloat {
        #if os(iOS) || os(visionOS)
        return largeControls ? 60 : 30
        #else
        return 30
        #endif
    }

    /// Extra trailing padding on visionOS to avoid overlapping the system copy-paste overlay
    private var visionOSTrailingPadding: CGFloat {
        #if os(visionOS)
        return sendButtonOnLeft ? 0 : 28
        #elseif os(iOS)
        if DeviceType.isRunningOnVisionPro && !sendButtonOnLeft { return 28 }
        return 0
        #else
        return 0
        #endif
    }

    private var micButtonSize: CGFloat {
        #if os(iOS) || os(visionOS)
        return largeControls ? 48 : 24
        #else
        return 24
        #endif
    }

    var body: some View {
        @Bindable var vm = chatVM

        ZStack(alignment: .top) {
            HStack(alignment: .center, spacing: 12) {
                if sendButtonOnLeft {
                    // Send/Stop button on left
                    sendButton

                    // Text field with optional mic button
                    textFieldWithOptionalMic
                } else {
                    // Text field with optional mic button
                    textFieldWithOptionalMic

                    // Send/Stop button on right (default)
                    sendButton
                }
            }
            .padding(.horizontal, 16)
            .padding(.trailing, visionOSTrailingPadding)
            .padding(.vertical, 12)
            .modifier(InputBarGlassModifier())
            .shadow(color: .black.opacity(0.2), radius: 12, y: 6)

            // Toast overlay - appears above input bar
            if showLoadingToast {
                LoadingToastView(message: "Model still loading...", icon: "hourglass")
                    .transition(.asymmetric(
                        insertion: .move(edge: .top).combined(with: .opacity),
                        removal: .opacity
                    ))
                    .offset(y: -50)
            } else if showNoModelToast {
                LoadingToastView(message: "No model loaded", icon: "cpu.fill")
                    .transition(.asymmetric(
                        insertion: .move(edge: .top).combined(with: .opacity),
                        removal: .opacity
                    ))
                    .offset(y: -50)
            } else if showMicErrorToast {
                LoadingToastView(message: micErrorText, icon: "mic.slash.fill")
                    .transition(.asymmetric(
                        insertion: .move(edge: .top).combined(with: .opacity),
                        removal: .opacity
                    ))
                    .offset(y: -50)
            }
        }
        // Update input text when speech recognition produces text
        // The recognizedText is updated continuously with the full transcription,
        // so we replace (not append) the speech portion each time
        .onChange(of: speechService.recognizedText) { _, newText in
            // Ignore recognizedText being cleared when speech stops — keep composed text
            guard isSpeechActive else { return }

            // Combine the text that existed before speech with the new recognized text
            if textBeforeSpeech.isEmpty {
                chatVM.inputText = newText
            } else if newText.isEmpty {
                chatVM.inputText = textBeforeSpeech
            } else {
                // Add space between existing text and speech if needed
                if textBeforeSpeech.hasSuffix(" ") || newText.hasPrefix(" ") {
                    chatVM.inputText = textBeforeSpeech + newText
                } else {
                    chatVM.inputText = textBeforeSpeech + " " + newText
                }
            }
        }
        // Track when speech starts/stops to manage text properly
        .onChange(of: speechService.isListening) { wasListening, isListening in
            if isListening && !wasListening {
                // Speech just started - save current input text and mark session active
                textBeforeSpeech = chatVM.inputText
                isSpeechActive = true
            } else if !isListening && wasListening {
                // Speech just stopped - deactivate session so recognizedText clearing is ignored
                isSpeechActive = false
                textBeforeSpeech = ""
            }
        }
        // Show speech service errors as toast (mic permission denied, etc.)
        .onChange(of: speechService.errorMessage) { _, newError in
            if let error = newError, !error.isEmpty {
                micErrorText = error
                withAnimation(.easeOut(duration: 0.2)) {
                    showMicErrorToast = true
                }
                Task {
                    try? await Task.sleep(for: .seconds(3))
                    withAnimation(.easeIn(duration: 0.3)) {
                        showMicErrorToast = false
                    }
                }
            }
        }
    }

    // MARK: - Text Field with Optional Mic

    @ViewBuilder
    private var textFieldWithOptionalMic: some View {
        if showMicrophone {
            HStack(spacing: 8) {
                textField
                microphoneButton
            }
        } else {
            textField
        }
    }

    // MARK: - Text Field

    private var textField: some View {
        @Bindable var vm = chatVM

        return TextField("Message...", text: $vm.inputText, axis: .vertical)
            .textFieldStyle(.plain)
            .lineLimit(1...6)
            .focused($isFocused)
            .disabled(chatVM.isGenerating || speechService.isListening)
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(inputFieldBackground)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .stroke(speechService.isListening ? Color.red.opacity(0.5) : inputFieldBorder, lineWidth: speechService.isListening ? 2 : 1)
            )
            .onSubmit {
                sendMessage()
            }
            .onKeyPress(.return, phases: .down) { press in
                if press.modifiers.contains(.shift) {
                    // Shift+Enter: insert newline (let system handle it)
                    return .ignored
                }
                sendMessage()
                return .handled
            }
    }

    // MARK: - Microphone Button

    private var microphoneButton: some View {
        Button {
            Task {
                await speechService.toggleListening()
            }
        } label: {
            ZStack {
                // Pulsing background when listening
                if speechService.isListening {
                    Circle()
                        .fill(Color.red.opacity(0.2))
                        .frame(width: micButtonSize * 1.3, height: micButtonSize * 1.3)
                        .modifier(PulseModifier())
                }

                Image(systemName: speechService.isListening ? "mic.fill" : "mic")
                    .font(.system(size: micButtonSize * 0.7, weight: .medium))
                    .foregroundStyle(speechService.isListening ? .red : .secondary)
                    .frame(width: micButtonSize, height: micButtonSize)
            }
        }
        .buttonStyle(.plain)
        .disabled(chatVM.isGenerating && !speechService.isListening)
        .help(speechService.isListening ? "Stop listening" : "Voice input")
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
            if chatVM.isGenerating {
                InferenceStopGlyph(
                    size: actionButtonSize,
                    isRotating: !isPrefill
                )
            } else {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: actionButtonSize * 0.9))
                    .foregroundStyle(buttonColor)
                    .frame(width: actionButtonSize, height: actionButtonSize)
            }
        }
        .buttonStyle(.plain)
        .disabled(!canSend && !chatVM.isGenerating)
        .keyboardShortcut(.return, modifiers: .command)
        .help(chatVM.isGenerating ? "Stop generation" : "Send message (⌘ Return)")
    }

    private var canSend: Bool {
        !chatVM.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !chatVM.isGenerating &&
        !modelManager.isLoadingModel &&
        modelManager.loadedModelId != nil
    }

    private var isPrefill: Bool {
        chatVM.isGenerating && chatVM.streamingContent.isEmpty
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
            showToast(loading: true)
            return
        }

        // Check if no model is loaded
        if modelManager.loadedModelId == nil {
            showToast(loading: false)
            return
        }

        guard canSend else { return }

        Task {
            await chatVM.sendMessage()
        }

        isFocused = false
    }

    private func showToast(loading: Bool) {
        withAnimation(.easeOut(duration: 0.2)) {
            if loading {
                showLoadingToast = true
            } else {
                showNoModelToast = true
            }
        }

        // Auto-dismiss after 2 seconds
        Task {
            try? await Task.sleep(for: .seconds(2))
            withAnimation(.easeIn(duration: 0.3)) {
                showLoadingToast = false
                showNoModelToast = false
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

// actionButtonSize moved to InputBar as computed property for largeControls support

// MARK: - Glass Effect Modifier (macOS 26+)

private struct InputBarGlassModifier: ViewModifier {
    func body(content: Content) -> some View {
        #if os(macOS)
        if #available(macOS 26.0, *) {
            content
                .glassEffect(.regular.interactive())
                .clipShape(RoundedRectangle(cornerRadius: 26, style: .continuous))
        } else {
            content
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 26, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 26, style: .continuous)
                        .stroke(inputBarBorder, lineWidth: 1)
                )
        }
        #else
        content
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 26, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 26, style: .continuous)
                    .stroke(inputBarBorder, lineWidth: 1)
            )
        #endif
    }
}

// MARK: - Stop Button Glyph

private struct InferenceStopGlyph: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var rotation: Double = 0
    @State private var secondaryRotation: Double = 0
    @State private var flicker = false
    @State private var pulse = false

    let size: CGFloat
    let isRotating: Bool

    private var flameGradient: Gradient {
        Gradient(stops: [
            .init(color: Color(red: 1.0, green: 0.82, blue: 0.35), location: 0.0),
            .init(color: Color(red: 1.0, green: 0.53, blue: 0.2), location: 0.35),
            .init(color: Color(red: 0.98, green: 0.2, blue: 0.25), location: 0.7),
            .init(color: Color(red: 1.0, green: 0.82, blue: 0.35), location: 1.0)
        ])
    }

    private var waitingIntensity: Double {
        isRotating ? 1.0 : 0.65
    }

    private var pulseScale: CGFloat {
        if reduceMotion || isRotating {
            return 1.0
        }
        return pulse ? 1.04 : 0.98
    }

    var body: some View {
        ZStack {
            // Subtle red core
            Circle()
                .fill(
                    RadialGradient(
                        gradient: Gradient(colors: [
                            Color.red.opacity(0.45),
                            Color.red.opacity(0.08)
                        ]),
                        center: .center,
                        startRadius: size * 0.12,
                        endRadius: size * 0.55
                    )
                )
                .frame(width: size, height: size)
                .opacity(waitingIntensity)

            // Main liquid ring
            Circle()
                .stroke(
                    AngularGradient(gradient: flameGradient, center: .center),
                    style: StrokeStyle(lineWidth: size * 0.18, lineCap: .round)
                )
                .rotationEffect(.degrees(rotation))
                .opacity(0.95 * waitingIntensity)
                .blur(radius: 0.5)

            // Secondary series ring (slightly offset)
            Circle()
                .trim(from: 0.12, to: 0.88)
                .stroke(
                    AngularGradient(gradient: flameGradient, center: .center),
                    style: StrokeStyle(lineWidth: size * 0.1, lineCap: .round)
                )
                .rotationEffect(.degrees(secondaryRotation))
                .opacity(0.65 * waitingIntensity)

            // Flame streaks that fade in/out
            ForEach(0..<3, id: \.self) { index in
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(red: 1.0, green: 0.84, blue: 0.35),
                                Color(red: 0.98, green: 0.2, blue: 0.25).opacity(0.05)
                            ],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .frame(width: size * 0.14, height: size * 0.36)
                    .offset(y: -size * 0.38)
                    .rotationEffect(.degrees(rotation + (Double(index) * 120)))
                    .opacity((flicker ? 0.9 : 0.25) * waitingIntensity)
                    .scaleEffect(flicker ? 1.0 : 0.7)
                    .blur(radius: 0.35)
                    .animation(
                        .easeInOut(duration: 0.85)
                            .repeatForever(autoreverses: true)
                            .delay(Double(index) * 0.18),
                        value: flicker
                    )
            }

            // Stop square
            RoundedRectangle(cornerRadius: size * 0.12, style: .continuous)
                .fill(Color.black.opacity(0.85))
                .frame(width: size * 0.32, height: size * 0.32)
                .shadow(color: .black.opacity(0.25), radius: 1, y: 0.5)
        }
        .frame(width: size, height: size)
        .scaleEffect(pulseScale)
        .onAppear {
            updateAnimations()
        }
        .onChange(of: isRotating) { _, _ in
            updateAnimations()
        }
        .onChange(of: reduceMotion) { _, _ in
            updateAnimations()
        }
    }

    private func updateAnimations() {
        if reduceMotion {
            rotation = 0
            secondaryRotation = 0
            flicker = false
            pulse = false
            return
        }

        if isRotating {
            pulse = false
            rotation = 0
            secondaryRotation = 0
            withAnimation(.linear(duration: 1.4).repeatForever(autoreverses: false)) {
                rotation = 360
            }
            withAnimation(.linear(duration: 2.2).repeatForever(autoreverses: false)) {
                secondaryRotation = -360
            }
            withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                flicker = true
            }
        } else {
            rotation = 0
            secondaryRotation = 0
            flicker = false
            pulse = false
            withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)) {
                pulse = true
            }
        }
    }
}

// MARK: - Loading Toast View

private struct LoadingToastView: View {
    let message: String
    var icon: String? = nil

    var body: some View {
        HStack(spacing: 8) {
            if let icon = icon {
                Image(systemName: icon)
                    .font(.subheadline)
                    .foregroundStyle(.orange)
            } else {
                ProgressView()
                    .controlSize(.small)
            }

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

// MARK: - Pulse Animation Modifier

private struct PulseModifier: ViewModifier {
    @State private var isPulsing = false

    func body(content: Content) -> some View {
        content
            .scaleEffect(isPulsing ? 1.2 : 0.9)
            .opacity(isPulsing ? 0.6 : 0.3)
            .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isPulsing)
            .onAppear {
                isPulsing = true
            }
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
