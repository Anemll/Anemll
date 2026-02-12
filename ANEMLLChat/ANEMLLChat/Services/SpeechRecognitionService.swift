//
//  SpeechRecognitionService.swift
//  ANEMLLChat
//
//  Speech-to-text using Apple's Speech framework
//

import Foundation
import Speech
import AVFoundation
#if os(macOS)
import AppKit
#endif

/// Service for speech recognition using Apple's Speech framework
@MainActor
@Observable
final class SpeechRecognitionService {
    static let shared = SpeechRecognitionService()

    // MARK: - State

    private(set) var isListening = false
    private(set) var isAuthorized = false
    private(set) var recognizedText = ""
    private(set) var errorMessage: String?

    // MARK: - Private

    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var audioEngine: AVAudioEngine?

    private init() {
        speechRecognizer = SFSpeechRecognizer(locale: Locale.current)
    }

    // MARK: - Authorization

    func requestAuthorization() async -> Bool {
        return await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                Task { @MainActor in
                    switch status {
                    case .authorized:
                        self.isAuthorized = true
                        self.errorMessage = nil
                    case .denied:
                        self.isAuthorized = false
                        self.errorMessage = "Speech recognition access denied"
                    case .restricted:
                        self.isAuthorized = false
                        self.errorMessage = "Speech recognition is restricted"
                    case .notDetermined:
                        self.isAuthorized = false
                        self.errorMessage = "Speech recognition not determined"
                    @unknown default:
                        self.isAuthorized = false
                        self.errorMessage = "Unknown authorization status"
                    }
                    continuation.resume(returning: self.isAuthorized)
                }
            }
        }
    }

    #if os(iOS)
    private func requestMicrophonePermission() async -> Bool {
        return await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }
    #else
    private func requestMicrophonePermission() async -> Bool {
        // On macOS, we need to use AVCaptureDevice for microphone permission
        // First check current status
        let status = AVCaptureDevice.authorizationStatus(for: .audio)

        switch status {
        case .authorized:
            return true
        case .notDetermined:
            // Request permission - this will show the system dialog
            let granted = await AVCaptureDevice.requestAccess(for: .audio)
            if !granted {
                logWarning("Microphone permission denied by user", category: .inference)
                errorMessage = "Microphone permission denied"
            }
            return granted
        case .denied:
            logWarning("Microphone permission denied. Opening System Settings > Privacy & Security > Microphone", category: .inference)
            errorMessage = "Mic denied — opening Privacy settings..."
            // Open System Settings directly to the microphone pane
            openMicrophoneSettings()
            return false
        case .restricted:
            logWarning("Microphone access is restricted on this device", category: .inference)
            errorMessage = "Microphone access is restricted"
            return false
        @unknown default:
            logWarning("Unknown microphone authorization status", category: .inference)
            return false
        }
    }

    /// Open System Settings to the microphone privacy pane
    private func openMicrophoneSettings() {
        // Try multiple URL schemes — macOS changes these between versions
        let urls = [
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
            "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_Microphone",
            "x-apple.systempreferences:com.apple.preference.security?Privacy"
        ]
        for urlString in urls {
            if let url = URL(string: urlString), NSWorkspace.shared.open(url) {
                logInfo("Opened System Settings via: \(urlString)", category: .inference)
                return
            }
        }
        // Last resort: open System Settings app
        NSWorkspace.shared.open(URL(fileURLWithPath: "/System/Applications/System Settings.app"))
    }
    #endif

    // MARK: - Recognition

    func startListening() async {
        // Check authorization - request if not already authorized
        if !isAuthorized {
            let authorized = await requestAuthorization()
            if !authorized {
                errorMessage = "Speech recognition not authorized"
                return
            }
        }

        // Check microphone permission
        let micPermission = await requestMicrophonePermission()
        if !micPermission {
            errorMessage = "Microphone access denied"
            return
        }

        guard let speechRecognizer = speechRecognizer, speechRecognizer.isAvailable else {
            errorMessage = "Speech recognizer not available"
            return
        }

        // Stop any existing recognition
        stopListening()

        // Clear previous text
        recognizedText = ""
        errorMessage = nil

        do {
            // Configure audio session
            #if os(iOS)
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
            #endif

            // Create recognition request
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            guard let recognitionRequest = recognitionRequest else {
                errorMessage = "Failed to create recognition request"
                return
            }

            recognitionRequest.shouldReportPartialResults = true

            // Use on-device recognition if available (iOS 13+)
            if #available(iOS 13, macOS 10.15, *) {
                recognitionRequest.requiresOnDeviceRecognition = speechRecognizer.supportsOnDeviceRecognition
            }

            // Create audio engine
            audioEngine = AVAudioEngine()
            guard let audioEngine = audioEngine else {
                errorMessage = "Failed to create audio engine"
                return
            }

            let inputNode = audioEngine.inputNode
            let recordingFormat = inputNode.outputFormat(forBus: 0)

            // Install tap on input node
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
                self?.recognitionRequest?.append(buffer)
            }

            // Start audio engine
            audioEngine.prepare()
            try audioEngine.start()

            isListening = true

            // Start recognition task
            recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
                Task { @MainActor in
                    guard let self = self else { return }

                    if let result = result {
                        self.recognizedText = result.bestTranscription.formattedString
                    }

                    if let error = error {
                        // Ignore cancellation errors
                        let nsError = error as NSError
                        if nsError.domain != "kAFAssistantErrorDomain" || nsError.code != 216 {
                            self.errorMessage = error.localizedDescription
                        }
                        self.stopListening()
                    }

                    if result?.isFinal == true {
                        self.stopListening()
                    }
                }
            }

        } catch {
            errorMessage = "Failed to start recording: \(error.localizedDescription)"
            stopListening()
        }
    }

    func stopListening() {
        // Stop audio engine
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil

        // End recognition request
        recognitionRequest?.endAudio()
        recognitionRequest = nil

        // Cancel recognition task
        recognitionTask?.cancel()
        recognitionTask = nil

        isListening = false

        // Deactivate audio session on iOS
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        #endif
    }

    func toggleListening() async {
        if isListening {
            stopListening()
        } else {
            await startListening()
        }
    }
}
