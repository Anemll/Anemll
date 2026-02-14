//
//  ToastView.swift
//  ANEMLLChat
//
//  A non-intrusive toast notification for errors and messages
//

import SwiftUI

// MARK: - Toast Types

enum ToastType {
    case error
    case warning
    case success
    case info

    var icon: String {
        switch self {
        case .error: return "xmark.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        case .success: return "checkmark.circle.fill"
        case .info: return "info.circle.fill"
        }
    }

    var color: Color {
        switch self {
        case .error: return .red
        case .warning: return .orange
        case .success: return .green
        case .info: return .blue
        }
    }
}

// MARK: - Toast View

struct ToastView: View {
    let message: String
    let type: ToastType
    var onDismiss: (() -> Void)?

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: type.icon)
                .font(.system(size: 20))
                .foregroundStyle(type.color)

            Text(message)
                .font(.subheadline)
                .foregroundStyle(.primary)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)

            Spacer()

            Button {
                onDismiss?()
            } label: {
                Image(systemName: "xmark")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(platformSecondaryBackground))
                .shadow(color: .black.opacity(0.15), radius: 8, y: 4)
        )
        .padding(.horizontal)
    }
}

// MARK: - Toast Modifier

struct ToastModifier: ViewModifier {
    @Binding var message: String?
    var type: ToastType
    var duration: TimeInterval

    @State private var workItem: DispatchWorkItem?

    func body(content: Content) -> some View {
        content
            .overlay(alignment: .top) {
                if let message = message {
                    ToastView(message: message, type: type) {
                        withAnimation {
                            self.message = nil
                        }
                    }
                    .padding(.top, 8)
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .onAppear {
                        // Cancel any existing timer
                        workItem?.cancel()

                        // Create new timer
                        let task = DispatchWorkItem {
                            withAnimation {
                                self.message = nil
                            }
                        }
                        workItem = task

                        // Auto-dismiss after duration
                        DispatchQueue.main.asyncAfter(deadline: .now() + duration, execute: task)
                    }
                    .onDisappear {
                        workItem?.cancel()
                    }
                }
            }
            .animation(.spring(response: 0.3, dampingFraction: 0.8), value: message != nil)
    }
}

// MARK: - View Extension

extension View {
    func toast(_ message: Binding<String?>, type: ToastType = .error, duration: TimeInterval = 4) -> some View {
        modifier(ToastModifier(message: message, type: type, duration: duration))
    }

    func errorToast(_ message: Binding<String?>, duration: TimeInterval = 5) -> some View {
        modifier(ToastModifier(message: message, type: .error, duration: duration))
    }

    func successToast(_ message: Binding<String?>, duration: TimeInterval = 3) -> some View {
        modifier(ToastModifier(message: message, type: .success, duration: duration))
    }
}

// MARK: - Platform Colors

#if os(iOS)
private let platformSecondaryBackground = UIColor.secondarySystemBackground
#else
private let platformSecondaryBackground = NSColor.controlBackgroundColor
#endif

// MARK: - Preview

#Preview {
    VStack {
        Spacer()
        Text("Main Content")
        Spacer()
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .toast(.constant("This is an error message that can be quite long and will wrap to multiple lines if needed"), type: .error)
}
