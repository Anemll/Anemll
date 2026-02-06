//
//  DebugInferenceOptions.swift
//  ANEMLLChat
//
//  Compile-time gates for debug-only inference controls
//

import Foundation

enum DebugInferenceOptions {
    /// Controls whether debug-only inference toggles are compiled in and visible.
    /// Use `-DDISABLE_DEBUG_INFERENCE_OPTIONS` to hard-disable in Debug builds.
    static let isEnabled: Bool = {
        #if DEBUG && !DISABLE_DEBUG_INFERENCE_OPTIONS
        return true
        #else
        return false
        #endif
    }()
}
