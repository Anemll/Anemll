//
//  ChatView.swift
//  ANEMLLChat
//
//  Main chat interface
//

import SwiftUI

struct ChatView: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(ModelManagerViewModel.self) private var modelManager

    @State private var scrollProxy: ScrollViewProxy?
    @State private var scrollMode: ScrollMode = .manual
    @State private var showScrollToBottom = false
    @State private var autoScrollTask: Task<Void, Never>?
    @State private var lastAutoScrollTime: Date = .distantPast
    @State private var inputAccessoryHeight: CGFloat = 0
    @State private var hasContentBelow = false  // True when content extends below visible area

    private let autoScrollInterval: TimeInterval = 0.07
    private let bottomVisibilityPadding: CGFloat = 4
    private let topFadeHeight: CGFloat = 72
    private let bottomScrimExtra: CGFloat = 56

    private var contentBottomPadding: CGFloat {
        max(24, inputAccessoryHeight + 64)  // Extra padding to keep content above scrim
    }

    private var scrollButtonBottomPadding: CGFloat {
        max(24, inputAccessoryHeight + 12)
    }
    
    private var bottomScrimHeight: CGFloat {
        max(48, inputAccessoryHeight + bottomScrimExtra)
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            // Messages
            messagesView

            VStack(spacing: 8) {
                // Model loading indicator
                if let progress = modelManager.loadingProgress, modelManager.isLoadingModel {
                    ModelLoadingBar(progress: progress)
                }

                // Input bar
                InputBar()
                    .environment(chatVM)
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 8)
            .background(
                GeometryReader { geometry in
                    Color.clear.preference(key: InputAccessoryHeightPreferenceKey.self, value: geometry.size.height)
                }
            )

            // Scroll to bottom button (centered horizontally, above input bar)
            if showScrollToBottom {
                Button {
                    setScrollMode(.follow)
                    scheduleAutoScroll(force: true)
                } label: {
                    Image(systemName: "chevron.down")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundStyle(.white)
                        .frame(width: 40, height: 40)
                        .background(.thinMaterial, in: Circle())  // Glass effect
                        .overlay(Circle().stroke(Color.white.opacity(0.2), lineWidth: 0.5))
                }
                .frame(maxWidth: .infinity, alignment: .center)  // Centered horizontally
                .padding(.bottom, max(90, inputAccessoryHeight + 40))  // Above input bar
            }
        }
        .navigationTitle(chatVM.currentConversation?.title ?? "Chat")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar(.hidden, for: .navigationBar)
        #endif
        // Model selector is in ContentView's detailToolbar - no need to duplicate here
        // Error toast (non-intrusive)
        .errorToast(Binding(
            get: { chatVM.errorMessage },
            set: { chatVM.errorMessage = $0 }
        ))
        .onPreferenceChange(InputAccessoryHeightPreferenceKey.self) { height in
            inputAccessoryHeight = height
        }
        .onChange(of: chatVM.currentConversation?.id) { _, _ in
            setScrollMode(.manual)
            hasContentBelow = false  // Reset when switching conversations
        }
        .onChange(of: hasContentBelow) { _, _ in
            updateChevronVisibility()
        }
        .onAppear {
            setScrollMode(.manual)
        }
    }

    // MARK: - Messages View

    private var messagesView: some View {
        ZStack(alignment: .bottom) {
            ScrollViewReader { proxy in
                ScrollView {
                        LazyVStack(spacing: 14) {
                            ForEach(visibleMessages) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }

                            // Streaming message with cursor (ChatGPT-like)
                            if chatVM.isGenerating {
                                StreamingMessageView(content: chatVM.streamingContent)
                                    .id("streaming")
                            }

                            // Bottom anchor for scrolling - needs height to ensure full scroll
                            Color.clear
                                .frame(height: 100)
                                .id("bottom")
                        }
                        .padding(.horizontal, 18)
                        .padding(.top, 16)
                        .padding(.bottom, contentBottomPadding)
                    }
                    .mask(topFadeMask)
                    .simultaneousGesture(
                        DragGesture(minimumDistance: 2)
                            .onChanged { _ in
                                setScrollMode(.manual)
                                // hasContentBelow is now updated by onScrollGeometryChange
                            }
                    )
                    .onAppear {
                        scrollProxy = proxy
                    }
                    .onChange(of: chatVM.currentConversation?.messages.count) { _, _ in
                        if scrollMode == .follow {
                            scheduleAutoScroll()
                        }
                    }
                    .onChange(of: chatVM.streamingContent) { _, _ in
                        if scrollMode == .follow {
                            scheduleAutoScroll()
                        }
                    }
                    // iOS 18+: Detect when content extends below visible area
                    .onScrollGeometryChange(for: Bool.self) { geometry in
                        // Content is below visible if contentSize.height > visibleRect.maxY + threshold
                        // Use larger threshold (80pt) so chevron hides when "close enough" to bottom
                        let threshold: CGFloat = 80
                        let contentBelow = geometry.contentSize.height > geometry.visibleRect.maxY + threshold
                        print("[ScrollGeo] contentH=\(Int(geometry.contentSize.height)) visibleMaxY=\(Int(geometry.visibleRect.maxY)) below=\(contentBelow)")
                        return contentBelow
                    } action: { oldValue, newValue in
                        if oldValue != newValue {
                            hasContentBelow = newValue
                        }
                    }
                }

            bottomScrim
            // Note: Scroll button moved to main body ZStack for proper layering
        }
        .background(chatBackground)
        .animation(.easeOut(duration: 0.25), value: showScrollToBottom)
    }

    private struct InputAccessoryHeightPreferenceKey: PreferenceKey {
        static var defaultValue: CGFloat = 0
        static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
            value = nextValue()
        }
    }

    private var topFadeMask: some View {
        GeometryReader { proxy in
            let height = max(1, proxy.size.height)
            let fadeFraction = min(0.22, max(0.08, topFadeHeight / height))
            LinearGradient(
                gradient: Gradient(stops: [
                    .init(color: .clear, location: 0),
                    .init(color: .black, location: fadeFraction),
                    .init(color: .black, location: 1)
                ]),
                startPoint: .top,
                endPoint: .bottom
            )
        }
    }

    private var bottomScrim: some View {
        LinearGradient(
            colors: [
                Color.black.opacity(0.0),
                Color.black.opacity(0.55),
                Color.black.opacity(0.92)
            ],
            startPoint: .top,
            endPoint: .bottom
        )
        .frame(height: bottomScrimHeight)
        .frame(maxWidth: .infinity)
        .allowsHitTesting(false)
    }

    private var visibleMessages: [ChatMessage] {
        var messages = chatVM.currentConversation?.messages.filter { $0.role != .system } ?? []
        // Avoid duplicating the streaming assistant message: we render it separately while generating.
        if chatVM.isGenerating, let last = messages.last, last.role == .assistant, !last.isComplete {
            messages.removeLast()
        }
        return messages
    }

    private var typingIndicator: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle()
                    .fill(Color.secondary)
                    .frame(width: 6, height: 6)
                    .scaleEffect(chatVM.isGenerating ? 1.0 : 0.5)
                    .animation(
                        .easeInOut(duration: 0.5)
                        .repeatForever()
                        .delay(Double(i) * 0.2),
                        value: chatVM.isGenerating
                    )
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(platformSecondaryBackground), in: Capsule())
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private enum ScrollMode {
        case manual
        case follow
    }

    private func updateChevronVisibility() {
        // Show chevron when content extends below visible area AND has messages
        let shouldShow = hasContentBelow && visibleMessages.count >= 2
        print("[ChevronV3] hasContentBelow=\(hasContentBelow) msgs=\(visibleMessages.count) shouldShow=\(shouldShow)")

        if showScrollToBottom != shouldShow {
            showScrollToBottom = shouldShow
        }
    }

    private func setScrollMode(_ mode: ScrollMode) {
        if scrollMode != mode {
            scrollMode = mode
        }

        if mode == .manual {
            autoScrollTask?.cancel()
            autoScrollTask = nil
        }
    }

    private func scheduleAutoScroll(force: Bool = false) {
        let now = Date()
        let elapsed = now.timeIntervalSince(lastAutoScrollTime)

        if force || elapsed >= autoScrollInterval {
            lastAutoScrollTime = now
            scrollToBottom(animated: true)
            return
        }

        autoScrollTask?.cancel()
        let delayMs = max(1, Int((autoScrollInterval - elapsed) * 1000))
        autoScrollTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(delayMs))
            lastAutoScrollTime = Date()
            scrollToBottom(animated: true)
        }
    }

    private func scrollToBottom(animated: Bool) {
        // hasContentBelow will be updated automatically by onScrollGeometryChange after scroll completes
        // Use .bottom anchor to ensure full scroll to the absolute bottom
        if animated {
            withAnimation(.easeInOut(duration: 0.35)) {
                scrollProxy?.scrollTo("bottom", anchor: .bottom)
            }
        } else {
            scrollProxy?.scrollTo("bottom", anchor: .bottom)
        }
    }
}

// MARK: - Streaming Message View (ChatGPT-like)

struct StreamingMessageView: View {
    let content: String
    @State private var cursorVisible = true

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            RoundedRectangle(cornerRadius: 2, style: .continuous)
                .fill(Color.secondary.opacity(0.45))
                .frame(width: 3)

            HStack(alignment: .bottom, spacing: 0) {
                if content.isEmpty {
                    // Show thinking indicator when no content yet
                    thinkingDots
                } else {
                    // Show streaming text with cursor
                    Text(content)
                        .textSelection(.enabled)
                        .lineSpacing(3)

                    // Blinking cursor
                    Text("|")
                        .fontWeight(.light)
                        .opacity(cursorVisible ? 1 : 0)
                        .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: cursorVisible)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .onAppear {
            cursorVisible = true
        }
    }

    private var thinkingDots: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .fill(Color.secondary)
                    .frame(width: 6, height: 6)
                    .opacity(0.7)
            }
        }
        .modifier(PulseAnimation())
    }
}

// Pulse animation for thinking dots
struct PulseAnimation: ViewModifier {
    @State private var isAnimating = false

    func body(content: Content) -> some View {
        content
            .scaleEffect(isAnimating ? 1.1 : 0.9)
            .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true), value: isAnimating)
            .onAppear {
                isAnimating = true
            }
    }
}

// MARK: - Model Loading Bar

struct ModelLoadingBar: View {
    let progress: ModelLoadingProgress

    @State private var startTime: Date?
    @State private var lastPercentage: Double = 0

    private var estimatedSecondsRemaining: Int? {
        guard let start = startTime,
              progress.percentage > 0.05 else { return nil } // Need at least 5% to estimate

        let elapsed = Date().timeIntervalSince(start)
        let progressRate = progress.percentage / elapsed
        guard progressRate > 0 else { return nil }

        let remaining = (1.0 - progress.percentage) / progressRate
        return max(1, Int(remaining))
    }

    private var etaString: String? {
        guard let seconds = estimatedSecondsRemaining else { return nil }
        if seconds < 60 {
            return "\(seconds)s"
        } else {
            let minutes = seconds / 60
            let secs = seconds % 60
            return "\(minutes)m \(secs)s"
        }
    }

    var body: some View {
        VStack(spacing: 4) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(progress.stage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    // Only show detail if it's not a file path (hide technical paths from users)
                    if let detail = progress.detail, !detail.isEmpty, !detail.contains("/") {
                        Text(detail)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }

                Spacer()

                HStack(spacing: 8) {
                    if let eta = etaString {
                        Text(eta)
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }

                    Text("\(Int(progress.percentage * 100))%")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundStyle(.secondary)
                }
            }

            ProgressView(value: progress.percentage)
                .progressViewStyle(.linear)
                .tint(.green)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(modelLoadingBackground)
        .onAppear {
            if startTime == nil {
                startTime = Date()
            }
        }
        .onChange(of: progress.percentage) { oldValue, newValue in
            // Reset timer if progress restarts
            if newValue < oldValue - 0.1 {
                startTime = Date()
            }
        }
    }
}

// MARK: - Platform Colors

#if os(iOS)
private let chatBackground = LinearGradient(
    colors: [
        Color(red: 0.06, green: 0.07, blue: 0.08),
        Color(red: 0.03, green: 0.03, blue: 0.04)
    ],
    startPoint: .topLeading,
    endPoint: .bottomTrailing
)
private let platformSecondaryBackground = UIColor.secondarySystemBackground
private let modelLoadingBackground = Color.white.opacity(0.06)
#else
private let platformBackground = NSColor.windowBackgroundColor
private let chatBackground = Color(platformBackground)
private let platformSecondaryBackground = NSColor.controlBackgroundColor
private let modelLoadingBackground = Color(platformSecondaryBackground)
#endif

#Preview {
    NavigationStack {
        ChatView()
            .environment(ChatViewModel())
            .environment(ModelManagerViewModel())
    }
}
