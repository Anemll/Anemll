//
//  View+Selectability.swift
//  ANEMLLChat
//
//  Conditional text selection helper
//

import SwiftUI

extension View {
    @ViewBuilder
    func selectable(_ enabled: Bool) -> some View {
        #if os(tvOS)
        self
        #else
        if enabled {
            self.textSelection(.enabled)
        } else {
            self.textSelection(.disabled)
        }
        #endif
    }
}
