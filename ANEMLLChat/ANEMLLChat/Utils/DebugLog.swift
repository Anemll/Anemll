//
//  DebugLog.swift
//  ANEMLLChat
//
//  Simple file-based debug logging for debugging GUI issues
//

import Foundation

/// Simple debug logger that writes to /tmp/anemll_debug.log
enum DebugLog {
    private static let logPath = "/tmp/anemll_debug.log"
    private static let lock = NSLock()

    /// Clear the log file
    static func clear() {
        lock.lock()
        defer { lock.unlock() }
        try? "".write(toFile: logPath, atomically: true, encoding: .utf8)
    }

    /// Write a message to the log file
    static func log(_ message: String, file: String = #file, line: Int = #line) {
        lock.lock()
        defer { lock.unlock() }

        let timestamp = ISO8601DateFormatter().string(from: Date())
        let filename = (file as NSString).lastPathComponent
        let logLine = "[\(timestamp)] [\(filename):\(line)] \(message)\n"

        if let handle = FileHandle(forWritingAtPath: logPath) {
            handle.seekToEndOfFile()
            if let data = logLine.data(using: .utf8) {
                handle.write(data)
            }
            handle.closeFile()
        } else {
            // Create file if doesn't exist
            try? logLine.write(toFile: logPath, atomically: true, encoding: .utf8)
        }
    }
}
