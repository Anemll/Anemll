//
//  ChatView.swift
//  ANEMLLChat
//
//  Main chat interface
//

import SwiftUI
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

struct ChatView: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(ModelManagerViewModel.self) private var modelManager

    @State private var scrollProxy: ScrollViewProxy?
    @State private var scrollMode: ScrollMode = .manual
    @State private var autoFollowSuspendedByUser = false
    @State private var showScrollToBottom = false
    @State private var autoScrollTask: Task<Void, Never>?
    @State private var lastAutoScrollTime: Date = .distantPast
    @State private var inputAccessoryHeight: CGFloat = 0
    @State private var hasContentBelow = false  // True when content extends below visible area
    @State private var wasAtBottomBeforeSend = false  // Track if user was at bottom when sending
    @State private var hasScrolledToQuestion = false  // Track if we've done the initial scroll to question
    @State private var userInteractedDuringGeneration = false
    @State private var streamingBuffer = StreamingBuffer()
    @State private var isUserDragging = false
    @State private var geometryBuffer = GeometryBuffer()
    @State private var scrollIdleTask: Task<Void, Never>?
    @State private var uiPauseLockedByUser = false
    @State private var isProgrammaticScroll = false
    @State private var programmaticScrollTask: Task<Void, Never>?

    private let autoScrollInterval: TimeInterval = 0.07
    private let bottomVisibilityPadding: CGFloat = 12
    private let scrollIdleDelayMs: Int = 180
    private let topFadeHeight: CGFloat = 72
    private let bottomScrimExtra: CGFloat = 56

    private var contentBottomPadding: CGFloat {
        max(24, inputAccessoryHeight + 48)  // Padding to keep content above input box with some clearance
    }

    private var typingPeekHeight: CGFloat {
        #if os(iOS)
        UIFont.preferredFont(forTextStyle: .body).lineHeight
        #else
        let font = NSFont.systemFont(ofSize: NSFont.systemFontSize)
        return font.ascender - font.descender + font.leading
        #endif
    }

    private var bottomScrollExtra: CGFloat {
        max(12, typingPeekHeight + 8)
    }

    private var scrollButtonBottomPadding: CGFloat {
        max(24, inputAccessoryHeight + 12)
    }
    
    private var bottomScrimHeight: CGFloat {
        max(48, inputAccessoryHeight + bottomScrimExtra)
    }

    private var uiTraceEnabled: Bool {
        chatVM.debugLevel >= 2
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
                    autoFollowSuspendedByUser = false
                    setScrollMode(.follow)
                    unlockUIUpdatesFromUser()
                    scrollToBottom(animated: true, toAbsoluteBottom: true)
                    // Update chevron visibility after scroll
                    Task { @MainActor in
                        try? await Task.sleep(for: .milliseconds(400))
                        updateChevronVisibility()
                    }
                } label: {
                    Image(systemName: "chevron.down")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundStyle(.primary)
                        .frame(width: 40, height: 40)
                        .modifier(ScrollButtonGlassModifier())
                }
                .buttonStyle(.plain)
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
            autoFollowSuspendedByUser = false
            // Don't reset hasContentBelow - let layout preferences determine it
            // Force chevron visibility check after content loads
            Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(200))
                updateChevronVisibility()
            }
        }
        .onChange(of: chatVM.pendingScrollToBottomRequest) { _, requestId in
            guard requestId != nil else { return }
            scrollToBottom(animated: true, toAbsoluteBottom: true)
        }
        .onChange(of: hasContentBelow) { _, _ in
            updateChevronVisibility()
        }
        .onChange(of: chatVM.isGenerating) { _, isGenerating in
            if isGenerating {
                // When generation starts:
                // 1. Remember if user was at/near bottom
                // 2. Set manual mode (no auto-scroll during generation)
                // 3. Scroll up just enough to show the loading dots below user's question
                wasAtBottomBeforeSend = !hasContentBelow
                hasScrolledToQuestion = false
                userInteractedDuringGeneration = false
                setScrollMode(.manual)

                // If user was at bottom, scroll up just enough to show the dots
                if wasAtBottomBeforeSend {
                    // Hide chevron immediately - we're at/near bottom
                    showScrollToBottom = false

                    // Scroll to show the streaming indicator (dots) just below the user's question
                    // This gives user feedback that generation has started
                    Task { @MainActor in
                        await Task.yield()
                        if !userInteractedDuringGeneration && chatVM.pendingScrollToBottomRequest == nil {
                            withAnimation(.easeOut(duration: 0.15)) {
                                // Scroll to typing peek so dots + prompt are visible
                                beginProgrammaticScroll()
                                scrollProxy?.scrollTo("typing-peek", anchor: .bottom)
                            }
                        }
                    }
                }
            } else {
                // When generation ends:
                // - Reset tracking state
                // - Scroll to last message to ensure proper geometry detection
                // - Update chevron visibility
                unlockUIUpdatesFromUser()
                hasScrolledToQuestion = false

                // Scroll to the completed message to update geometry
                Task { @MainActor in
                    try? await Task.sleep(for: .milliseconds(100))
                    if let lastMsg = visibleMessages.last {
                        withAnimation(.easeOut(duration: 0.2)) {
                            beginProgrammaticScroll()
                            scrollProxy?.scrollTo(lastMsg.id, anchor: .bottom)
                        }
                    }
                    // Give scroll geometry time to update
                    try? await Task.sleep(for: .milliseconds(150))
                    updateChevronVisibility()

                }
            }
        }
        .onChange(of: chatVM.streamingContent) { _, content in
            bufferStreamingChange(content)
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
                                    .equatable()
                                    .id(message.id)
                            }

                            // Streaming message with cursor (ChatGPT-like)
                            if chatVM.isGenerating {
                                StreamingMessageView(content: chatVM.streamingContent)
                                    .id("streaming")

                                if chatVM.streamingContent.isEmpty {
                                    Color.clear
                                        .frame(height: typingPeekHeight)
                                        .id("typing-peek")
                                }
                            }

                            // Bottom spacer - anchor point for scroll
                            Color.clear
                                .frame(height: contentBottomPadding + bottomScrollExtra)
                                .id("bottom")
                        }
                        .padding(.horizontal, 18)
                        .padding(.top, 16)
                    }
                    .mask(topFadeMask)
                    .simultaneousGesture(
                        DragGesture(minimumDistance: 2)
                            .onChanged { value in
                                if uiTraceEnabled && !isUserDragging {
                                    isUserDragging = true
                                    logDebug("[Scroll] drag begin generating=\(chatVM.isGenerating) follow=\(scrollMode == .follow) hasBelow=\(hasContentBelow)", category: .ui)
                                }
                                markScrollActivity()

                                if chatVM.isGenerating {
                                    userInteractedDuringGeneration = true
                                    showScrollToBottom = true
                                }

                                // On iOS, dragging down scrolls "up" (toward older messages).
                                let isScrollingUp = value.translation.height > 0
                                if scrollMode == .follow && isScrollingUp {
                                    pauseAutoFollowDueToUserScroll()
                                }
                            }
                            .onEnded { _ in
                                if uiTraceEnabled && isUserDragging {
                                    logDebug("[Scroll] drag end", category: .ui)
                                }
                                isUserDragging = false
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
                    .onScrollGeometryChange(for: Double.self) { geometry in
                        Double(geometry.visibleRect.minY)
                    } action: { oldValue, newValue in
                        if chatVM.isGenerating && abs(newValue - oldValue) > 0.5 {
                            markScrollActivity()
                        }

                        if isProgrammaticScroll { return }

                        // In scroll coordinates, scrolling "up" decreases `visibleRect.minY`.
                        let delta = newValue - oldValue
                        guard delta < -0.5 else { return }

                        Task { @MainActor in
                            await Task.yield()
                            pauseAutoFollowDueToUserScroll()
                        }
                    }
                    .onScrollGeometryChange(for: CGFloat.self) { geometry in
                        geometry.contentSize.height - geometry.visibleRect.maxY
                    } action: { oldValue, newValue in
                        let oldHasBelow = oldValue > bottomVisibilityPadding
                        let newHasBelow = newValue > bottomVisibilityPadding

                        if chatVM.isGenerating {
                            if !userInteractedDuringGeneration && oldHasBelow != newHasBelow {
                                userInteractedDuringGeneration = true
                                showScrollToBottom = true
                            }
                            return
                        }

                        guard hasContentBelow != newHasBelow else { return }

                        geometryBuffer.latest = newHasBelow
                        guard !geometryBuffer.isScheduled else { return }
                        geometryBuffer.isScheduled = true

                        Task { @MainActor in
                            await Task.yield()
                            if let pending = geometryBuffer.latest, hasContentBelow != pending {
                                if uiTraceEnabled {
                                    logDebug("[ScrollGeo] hasBelow=\(pending)", category: .ui)
                                }
                                hasContentBelow = pending
                            }
                            geometryBuffer.isScheduled = false
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

    private final class StreamingBuffer {
        var latest: String = ""
        var isScheduled = false
    }
    
    private final class GeometryBuffer {
        var latest: Bool?
        var isScheduled = false
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

    private func markScrollActivity() {
        guard chatVM.isGenerating else { return }
        guard !uiPauseLockedByUser else { return }

        if !chatVM.uiUpdatesPaused {
            chatVM.setUIUpdatesPaused(true)
            if uiTraceEnabled {
                logDebug("[UI Pause] begin", category: .ui)
            }
        }

        scrollIdleTask?.cancel()
        scrollIdleTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(scrollIdleDelayMs))
            chatVM.setUIUpdatesPaused(false)
            if uiTraceEnabled {
                logDebug("[UI Pause] end", category: .ui)
            }
        }
    }

    private func pauseAutoFollowDueToUserScroll() {
        autoFollowSuspendedByUser = true
        showScrollToBottom = true
        if scrollMode == .follow {
            setScrollMode(.manual)
        }

        if chatVM.isGenerating {
            userInteractedDuringGeneration = true
            lockUIUpdatesForUserScroll()
        }
    }

    private func lockUIUpdatesForUserScroll() {
        guard chatVM.isGenerating else { return }
        guard !uiPauseLockedByUser else { return }
        uiPauseLockedByUser = true
        chatVM.setUIUpdatesPaused(true)
        if uiTraceEnabled {
            logDebug("[UI Pause] locked by user scroll", category: .ui)
        }
    }

    private func unlockUIUpdatesFromUser() {
        guard uiPauseLockedByUser else { return }
        uiPauseLockedByUser = false
        chatVM.setUIUpdatesPaused(false)
        if uiTraceEnabled {
            logDebug("[UI Pause] unlocked", category: .ui)
        }
    }

    private func beginProgrammaticScroll() {
        isProgrammaticScroll = true
        programmaticScrollTask?.cancel()
        programmaticScrollTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(200))
            isProgrammaticScroll = false
        }
    }

    private func updateChevronVisibility() {
        let hasContent = visibleMessages.count >= 1 || chatVM.isGenerating

        if autoFollowSuspendedByUser {
            if showScrollToBottom != hasContent {
                showScrollToBottom = hasContent
            }
            return
        }

        if scrollMode == .follow {
            if showScrollToBottom {
                showScrollToBottom = false
            }
            return
        }

        if chatVM.isGenerating && userInteractedDuringGeneration {
            showScrollToBottom = true
            return
        }

        // Show chevron ONLY when content actually extends below visible area
        // We no longer show it just because we're in manual mode during generation
        // The geometry-based hasContentBelow is the source of truth
        let shouldShow = hasContentBelow && hasContent

        // Debug logging (can be removed in production)
        // print("[ChevronV5] hasContentBelow=\(hasContentBelow) msgs=\(visibleMessages.count) generating=\(chatVM.isGenerating) shouldShow=\(shouldShow)")

        if showScrollToBottom != shouldShow {
            if uiTraceEnabled {
                logDebug("[Chevron] show=\(shouldShow) hasBelow=\(hasContentBelow) generating=\(chatVM.isGenerating)", category: .ui)
            }
            showScrollToBottom = shouldShow
        }
    }

    private func bufferStreamingChange(_ content: String) {
        streamingBuffer.latest = content

        guard !streamingBuffer.isScheduled else { return }
        streamingBuffer.isScheduled = true

        Task { @MainActor in
            await Task.yield()
            streamingBuffer.isScheduled = false
            processStreamingChange(streamingBuffer.latest)
        }
    }

    private func processStreamingChange(_ content: String) {
        // During streaming:
        // 1. When first tokens arrive (>20 chars) and user was at bottom - scroll to position question at top
        // 2. Then let content fill downward (no more auto-scroll)
        // 3. Update chevron visibility as content grows
        if chatVM.isGenerating && !content.isEmpty {
            // First tokens arrived - scroll so user's question is at the TOP of viewport
            // Wait for at least ~20 chars so there's visible content before scrolling
            if wasAtBottomBeforeSend && !hasScrolledToQuestion && !userInteractedDuringGeneration && content.count > 20 {
                hasScrolledToQuestion = true
                // Find the last user message ID to scroll to
                if let lastUserMessage = visibleMessages.last(where: { $0.role == .user }) {
                    Task { @MainActor in
                        try? await Task.sleep(for: .milliseconds(50))
                        if !userInteractedDuringGeneration {
                            withAnimation(.easeOut(duration: 0.3)) {
                                // Scroll user's question to TOP of viewport (anchor: .top)
                                // This positions the question at the very top with answer filling below
                                beginProgrammaticScroll()
                                scrollProxy?.scrollTo(lastUserMessage.id, anchor: .top)
                            }
                        }
                    }
                }
            }

            // As content grows, check if it extends below visible area
            // During streaming, we assume content below if we've scrolled to question
            // and have substantial content (likely extends below fold)
            if hasScrolledToQuestion && content.count > 200 {
                // Content is likely below visible area, show chevron
                if !showScrollToBottom {
                    showScrollToBottom = true
                }
            }

            // Also always update chevron based on geometry
            updateChevronVisibility()
        }

        if scrollMode == .follow {
            scheduleAutoScroll()
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

    private func scrollToBottom(animated: Bool, toAbsoluteBottom: Bool = false) {
        // Always scroll to "bottom" spacer when user clicks chevron
        // This ensures we reach the absolute end of content
        let targetId: AnyHashable = "bottom"

        if animated {
            withAnimation(.easeInOut(duration: 0.35)) {
                beginProgrammaticScroll()
                scrollProxy?.scrollTo(targetId, anchor: .bottom)
            }
        } else {
            beginProgrammaticScroll()
            scrollProxy?.scrollTo(targetId, anchor: .bottom)
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

            VStack(alignment: .leading, spacing: 0) {
                if content.isEmpty {
                    // Show thinking indicator when no content yet
                    thinkingDots
                } else {
                    // Show streaming text with markdown rendering
                    HStack(alignment: .bottom, spacing: 0) {
                        MarkdownView(content: content, isUserMessage: false)

                        // Blinking cursor
                        Text("|")
                            .fontWeight(.light)
                            .opacity(cursorVisible ? 1 : 0)
                            .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: cursorVisible)
                    }
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

// MARK: - Glass Effect Modifier (macOS 26+)

private struct ScrollButtonGlassModifier: ViewModifier {
    func body(content: Content) -> some View {
        #if os(macOS)
        if #available(macOS 26.0, *) {
            content
                .glassEffect(.regular.interactive())
                .clipShape(Circle())
        } else {
            content
                .background(.thinMaterial, in: Circle())
                .overlay(Circle().stroke(Color.white.opacity(0.2), lineWidth: 0.5))
        }
        #else
        content
            .background(.thinMaterial, in: Circle())
            .overlay(Circle().stroke(Color.white.opacity(0.2), lineWidth: 0.5))
        #endif
    }
}

#Preview {
    NavigationStack {
        ChatView()
            .environment(ChatViewModel())
            .environment(ModelManagerViewModel())
    }
}
