//
//  UIFreezeWatchdog.swift
//  ANEMLLChat
//
//  Detects UI freezes and logs diagnostic information
//  macOS only - uses Process API not available on iOS
//

import Foundation

#if os(macOS)
import AppKit

/// Watchdog that detects when the main thread is blocked for too long
/// Only available on macOS (uses Process API for sampling)
final class UIFreezeWatchdog {
    static let shared = UIFreezeWatchdog()

    private var watchdogThread: Thread?
    private var isRunning = false
    private var lastPingTime: Date = Date()
    private var pingCount: UInt64 = 0
    private let pingInterval: TimeInterval = 0.1  // Check every 100ms
    private let freezeThreshold: TimeInterval = 0.5  // Alert if blocked > 500ms
    private var freezeStartTime: Date?
    private var lastStackTrace: String?

    private init() {}

    /// Start the watchdog if debug level >= 2 (Verbose)
    func start() {
        // Only start if debug level is Verbose (2)
        let debugLevel = UserDefaults.standard.object(forKey: "debugLevel") as? Int ?? 0
        guard debugLevel >= 2 else {
            // Silently skip - watchdog is off by default
            return
        }

        guard !isRunning else { return }
        isRunning = true
        lastPingTime = Date()

        // Start watchdog thread
        watchdogThread = Thread { [weak self] in
            self?.watchdogLoop()
        }
        watchdogThread?.name = "UIFreezeWatchdog"
        watchdogThread?.qualityOfService = .userInteractive
        watchdogThread?.start()

        // Start main thread ping timer
        startMainThreadPinger()

        print("[Watchdog] Started - will detect UI freezes > \(Int(freezeThreshold * 1000))ms")
    }

    func stop() {
        isRunning = false
        watchdogThread?.cancel()
        watchdogThread = nil
        print("[Watchdog] Stopped")
    }

    private func startMainThreadPinger() {
        // Use a repeating timer on main thread to update lastPingTime
        DispatchQueue.main.async { [weak self] in
            self?.scheduleNextPing()
        }
    }

    private func scheduleNextPing() {
        guard isRunning else { return }

        // Update ping time
        lastPingTime = Date()
        pingCount += 1

        // Schedule next ping
        DispatchQueue.main.asyncAfter(deadline: .now() + pingInterval) { [weak self] in
            self?.scheduleNextPing()
        }
    }

    private func watchdogLoop() {
        while isRunning && !Thread.current.isCancelled {
            Thread.sleep(forTimeInterval: pingInterval)

            let now = Date()
            let timeSinceLastPing = now.timeIntervalSince(lastPingTime)

            if timeSinceLastPing > freezeThreshold {
                if freezeStartTime == nil {
                    // Freeze just started
                    freezeStartTime = lastPingTime
                    reportFreezeStart(duration: timeSinceLastPing)
                }
            } else {
                if let startTime = freezeStartTime {
                    // Freeze ended
                    let totalDuration = now.timeIntervalSince(startTime)
                    reportFreezeEnd(duration: totalDuration)
                    freezeStartTime = nil
                }
            }
        }
    }

    private func reportFreezeStart(duration: TimeInterval) {
        let durationMs = Int(duration * 1000)
        let pid = ProcessInfo.processInfo.processIdentifier

        // Auto-sample the process
        autoSampleProcess(pid: pid)

        print("""

        ⚠️ [FREEZE DETECTED] Main thread blocked for \(durationMs)ms
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Ping count: \(pingCount)
        Last ping: \(lastPingTime)
        PID: \(pid)

        Auto-sampling process... check /tmp/freeze_\(pid).txt
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        """)
    }

    private func reportFreezeEnd(duration: TimeInterval) {
        let durationMs = Int(duration * 1000)
        print("""

        ✅ [FREEZE ENDED] Total duration: \(durationMs)ms
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        """)
    }

    private func autoSampleProcess(pid: Int32) {
        // Run sample command in background to capture stack trace
        let outputFile = "/tmp/freeze_\(pid).txt"

        DispatchQueue.global(qos: .userInteractive).async {
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/sample")
            task.arguments = ["\(pid)", "1", "-file", outputFile]

            do {
                try task.run()
                task.waitUntilExit()

                if task.terminationStatus == 0 {
                    // Read and print key parts of the sample
                    if let content = try? String(contentsOfFile: outputFile, encoding: .utf8) {
                        let keyLines = self.extractKeyStackFrames(from: content)
                        print("""

                        📊 [STACK SAMPLE] Key frames from \(outputFile):
                        \(keyLines)

                        """)
                    }
                } else {
                    print("[Watchdog] Sample failed with status \(task.terminationStatus) - may need sudo")
                }
            } catch {
                print("[Watchdog] Failed to run sample: \(error)")
            }
        }
    }

    private func extractKeyStackFrames(from sampleOutput: String) -> String {
        // Extract the most relevant stack frames (SwiftUI, our app code)
        let lines = sampleOutput.components(separatedBy: "\n")
        var relevantLines: [String] = []
        var inMainThread = false
        var frameCount = 0

        for line in lines {
            // Find main thread section
            if line.contains("Main Thread") || line.contains("Thread 0x") && line.contains("main") {
                inMainThread = true
                relevantLines.append("--- Main Thread ---")
                continue
            }

            // Stop at next thread
            if inMainThread && line.contains("Thread 0x") && !line.contains("main") {
                break
            }

            if inMainThread {
                // Include frames with our app name, SwiftUI, or key system calls
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.contains("ANEMLLChat") ||
                   trimmed.contains("SwiftUI") ||
                   trimmed.contains("ScrollView") ||
                   trimmed.contains("LazyVStack") ||
                   trimmed.contains("ForEach") ||
                   trimmed.contains("layout") ||
                   trimmed.contains("Layout") ||
                   trimmed.contains("update") ||
                   trimmed.contains("render") {
                    relevantLines.append(trimmed)
                    frameCount += 1
                    if frameCount > 30 { break }  // Limit output
                }
            }
        }

        if relevantLines.isEmpty {
            return "[No relevant frames found - check full output at /tmp/freeze_*.txt]"
        }

        return relevantLines.joined(separator: "\n")
    }
}

// MARK: - Debug helpers

extension UIFreezeWatchdog {
    /// Call this from SwiftUI views to mark activity points
    func markActivity(_ label: String) {
        guard isRunning else { return }
        let timestamp = Date()
        print("[Watchdog] Activity: \(label) at \(timestamp)")
    }
}

#else
// iOS stub - watchdog not available
final class UIFreezeWatchdog {
    static let shared = UIFreezeWatchdog()
    private init() {}
    func start() {}
    func stop() {}
    func markActivity(_ label: String) {}
}
#endif
