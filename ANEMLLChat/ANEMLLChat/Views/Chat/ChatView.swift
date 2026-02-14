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
    @AppStorage("enableMarkup") private var enableMarkup: Bool = StorageService.defaultEnableMarkupValue

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
    @State private var isScrollMovementActive = false
    @State private var scrollMovementTask: Task<Void, Never>?
    @State private var lastUserScrollTime: Date = .distantPast
    @State private var lastScrollActivityTime: Date = .distantPast
    @State private var generationExtraPad: CGFloat = 0
    @State private var chevronBuffer = ChevronBuffer()
    @State private var cachedVisibleMessages: [ChatMessage] = []  // Cache to prevent repeated filtering

    private let autoScrollInterval: TimeInterval = 0.07
    private let bottomVisibilityPadding: CGFloat = 12
    private let scrollIdleDelayMs: Int = 180
    private let topFadeHeight: CGFloat = 72
    private let bottomScrimExtra: CGFloat = 56
    private let userScrollDetectionWindow: TimeInterval = 0.35
    private let sendScrollSettleDelayMs: Int = 120
    private let scrollMovementIdleMs: Int = 140
    private let questionScrollMinChars: Int = 160
    private let chevronSuspendWindow: TimeInterval = 0.3  // Short delay to avoid flicker during scroll

    private var generationExtraPadMax: CGFloat {
        max(72, typingPeekHeight * 4)
    }

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

    private var loadingGaugeBottomSpacing: CGFloat {
        max(24, inputAccessoryHeight + 24) * 3
    }

    private var uiTraceEnabled: Bool {
        chatVM.debugLevel >= 2
    }

    private var allowTextSelection: Bool {
        #if os(macOS)
        return !chatVM.isGenerating
        #else
        return true
        #endif
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            // Messages
            messagesView

            // Centered prompt when no model is loaded and not loading
            // Wait for initial startup (model list + auto-load) to finish before showing
            if modelManager.hasCompletedInitialLoad && modelManager.loadedModelId == nil && !modelManager.isLoadingModel {
                VStack(spacing: 16) {
                    Image(systemName: modelManager.errorMessage != nil ? "exclamationmark.triangle" : "cpu")
                        .font(.system(size: 40))
                        .foregroundStyle(modelManager.errorMessage != nil ? .red : .secondary)

                    if let error = modelManager.errorMessage {
                        Text("Model loading failed")
                            .font(.headline)
                            .foregroundStyle(.primary)
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                    }

                    Button {
                        modelManager.requestModelSelection = true
                    } label: {
                        Label("Download or Select Model", systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(true)
            }

            if let progress = modelManager.loadingProgress, modelManager.isLoadingModel {
                // Full-screen frosted glass overlay to dim text behind
                ZStack(alignment: .center) {
                    // Dark tint layer for better contrast
                    Color.black.opacity(0.5)
                        .ignoresSafeArea()

                    // Frosted glass effect on top
                    Rectangle()
                        .fill(.thickMaterial)
                        .opacity(0.9)
                        .ignoresSafeArea()

                    // Centered loading gauge
                    ModelLoadingGauge(progress: progress, modelName: modelManager.loadingModelName)
                        .fixedSize()  // Prevent gauge from expanding to fill space
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .transition(.opacity)
                .allowsHitTesting(false)
            }

            VStack(spacing: 8) {
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

            scrollToBottomOverlay
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
            generationExtraPad = 0
            updateCachedVisibleMessages()
            // Don't reset hasContentBelow - let layout preferences determine it
            // Force chevron visibility check after content loads
            Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(200))
                updateChevronVisibility()
            }
        }
        .onChange(of: chatVM.currentConversation?.messages.count) { _, _ in
            updateCachedVisibleMessages()
        }
        .onChange(of: chatVM.pendingScrollToBottomRequest) { _, requestId in
            guard requestId != nil else { return }
            autoFollowSuspendedByUser = false
            unlockUIUpdatesFromUser()
            forceScrollToLatestUserMessageTop()
        }
        // Removed onChange(of: hasContentBelow) - no longer using geometry-based detection
        .onChange(of: chatVM.isGenerating) { _, isGenerating in
            // Update cached messages when generation state changes
            updateCachedVisibleMessages()

            if isGenerating {
                // When generation starts:
                // 1. Remember if user was at/near bottom
                // 2. Set manual mode (no auto-scroll during generation)
                // 3. Scroll up just enough to show the loading dots below user's question
                wasAtBottomBeforeSend = !hasContentBelow
                hasScrolledToQuestion = false
                userInteractedDuringGeneration = false
                setScrollMode(.manual)
                generationExtraPad = wasAtBottomBeforeSend ? generationExtraPadMax : 0

                // If user was at bottom, keep focus near the latest message.
                if wasAtBottomBeforeSend {
                    // Hide chevron immediately - we're at/near bottom
                    showScrollToBottom = false
                }
            } else {
                // When generation ends:
                // - Reset tracking state only, no scroll operations to avoid layout issues
                unlockUIUpdatesFromUser()
                hasScrolledToQuestion = false
                generationExtraPad = 0

                // Show chevron after generation if response is likely below fold
                // Use a heuristic: if streaming content has multiple lines, show chevron
                let responseLines = chatVM.streamingContent.components(separatedBy: "\n").count
                let responseLength = chatVM.streamingContent.count
                if responseLines > 8 || responseLength > 400 {
                    // Response is long enough that user likely needs to scroll to see end
                    Task { @MainActor in
                        try? await Task.sleep(for: .milliseconds(300))
                        if !showScrollToBottom {
                            showScrollToBottom = true
                        }
                    }
                }
            }
        }
        // Auto-scroll during streaming when in follow mode (user clicked chevron)
        .onChange(of: chatVM.streamingContent) { _, newContent in
            if scrollMode == .follow && chatVM.isGenerating {
                scheduleAutoScroll()
            }

            // Show chevron during generation when content gets long (likely below fold)
            // This supplements the geometry-based detection which may not fire during streaming
            if chatVM.isGenerating && !showScrollToBottom && scrollMode != .follow {
                let lines = newContent.components(separatedBy: "\n").count
                let length = newContent.count
                // Heuristic: if content is substantial, show chevron
                if lines > 6 || length > 300 {
                    showScrollToBottom = true
                }
            }
        }
        .onAppear {
            setScrollMode(.manual)
            updateCachedVisibleMessages()
        }
    }

    // MARK: - Scroll To Bottom Overlay

    @ViewBuilder
    private var scrollToBottomOverlay: some View {
        // Scroll to bottom button (centered horizontally, above input bar)
        // Use opacity + animation instead of if/else for smooth fade
        Button {
            // Hide chevron immediately and scroll to bottom
            withAnimation(.easeOut(duration: 0.2)) {
                showScrollToBottom = false
            }
            autoFollowSuspendedByUser = false
            setScrollMode(.follow)
            unlockUIUpdatesFromUser()
            scrollToBottom(animated: true, toAbsoluteBottom: true)
        } label: {
            ZStack {
                Rectangle()
                    .fill(Color.clear)
                    .frame(width: 56, height: 56)

                Image(systemName: "arrow.down")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.primary)
                    .frame(width: 40, height: 40)
                    .modifier(ScrollButtonGlassModifier())
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        #if os(macOS)
        .pointingHandCursor()
        #endif
        .frame(maxWidth: .infinity, alignment: .center)  // Centered horizontally
        .padding(.bottom, max(90, inputAccessoryHeight + 40))  // Above input bar
        .opacity(showScrollToBottom ? 1 : 0)
        .animation(.easeInOut(duration: 0.25), value: showScrollToBottom)
        .allowsHitTesting(showScrollToBottom)  // Only clickable when visible
    }

    // MARK: - Messages View

    private var messagesView: some View {
        ZStack(alignment: .bottom) {
            ScrollViewReader { proxy in
                ScrollView {
                        VStack(spacing: 14) {
                            ForEach(visibleMessages) { message in
                                MessageBubble(message: message, allowSelection: allowTextSelection, enableMarkup: enableMarkup)
                                    .id(message.id)
                            }

                            // Streaming message with cursor (ChatGPT-like)
                            if chatVM.isGenerating {
                                StreamingMessageView(content: chatVM.streamingContent, allowSelection: allowTextSelection, enableMarkup: enableMarkup)
                                    .id("streaming")

                                if chatVM.streamingContent.isEmpty {
                                    Color.clear
                                        .frame(height: typingPeekHeight)
                                        .id("typing-peek")
                                }
                            }

                            // Bottom spacer - anchor point for scroll
                            // Also used to detect when bottom is visible (for chevron logic)
                            Color.clear
                                .frame(height: contentBottomPadding + bottomScrollExtra + generationExtraPad)
                                .id("bottom")
                                .background(
                                    GeometryReader { geo in
                                        // Bottom is visible when its top edge is within the visible screen area
                                        // (accounting for input bar height ~100px from bottom)
                                        let bottomY = geo.frame(in: .global).minY
                                        let screenHeight = UIScreen.mainHeight
                                        // If bottomY < screenHeight - 50, the bottom spacer is on screen (user at bottom)
                                        // If bottomY >= screenHeight - 50, there's content below the fold
                                        let isAtBottom = bottomY < screenHeight - 50
                                        Color.clear
                                            .preference(
                                                key: BottomVisiblePreferenceKey.self,
                                                value: isAtBottom
                                            )
                                    }
                                )
                        }
                        .padding(.horizontal, 18)
                        .padding(.top, 16)
                    }
                    #if os(macOS)
                    .overlay(topFadeOverlay, alignment: .top)
                    #else
                    .mask(topFadeMask)
                    #endif
                    .onAppear {
                        scrollProxy = proxy
                    }
                    .onChange(of: chatVM.currentConversation?.messages.count) { _, _ in
                        if scrollMode == .follow {
                            scheduleAutoScroll()
                        }
                    }
                    // Detect user scroll to interrupt auto-follow (all platforms)
                    .onScrollPhaseChange { oldPhase, newPhase in
                        if newPhase == .interacting || newPhase == .decelerating {
                            if scrollMode == .follow {
                                setScrollMode(.manual)
                                autoFollowSuspendedByUser = true
                            }
                        }
                    }
                    #if os(iOS)
                    // Also detect tap-to-stop on iOS (onScrollPhaseChange doesn't fire for taps)
                    .simultaneousGesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { _ in
                                if scrollMode == .follow && chatVM.isGenerating {
                                    setScrollMode(.manual)
                                    autoFollowSuspendedByUser = true
                                }
                            }
                    )
                    #endif
                }

            bottomScrim
            // Note: Scroll button moved to main body ZStack for proper layering
        }
        .background(chatBackground)
        .onPreferenceChange(BottomVisiblePreferenceKey.self) { isAtBottom in
            // Don't show chevron on empty chat
            guard visibleMessages.count > 0 || chatVM.isGenerating else {
                if showScrollToBottom {
                    showScrollToBottom = false
                }
                return
            }

            // isAtBottom = true means user can see the bottom (no content below fold)
            // isAtBottom = false means there's content below the fold (show chevron)

            if isAtBottom {
                // User is at bottom - hide chevron
                // But during generation in follow mode, keep it hidden (we're auto-scrolling)
                if showScrollToBottom && scrollMode == .follow {
                    // In follow mode at bottom - hide chevron
                    showScrollToBottom = false
                    autoFollowSuspendedByUser = false
                } else if showScrollToBottom && !chatVM.isGenerating {
                    // Not generating and at bottom - hide chevron
                    showScrollToBottom = false
                    autoFollowSuspendedByUser = false
                }
            } else {
                // Content below fold - show chevron (even during generation)
                if !showScrollToBottom {
                    showScrollToBottom = true
                }
            }
        }
    }

    private struct InputAccessoryHeightPreferenceKey: PreferenceKey {
        static var defaultValue: CGFloat = 0
        static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
            value = nextValue()
        }
    }

    private struct BottomVisiblePreferenceKey: PreferenceKey {
        static var defaultValue: Bool = true
        static func reduce(value: inout Bool, nextValue: () -> Bool) {
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

    #if os(macOS)
    private var topFadeOverlay: some View {
        LinearGradient(
            gradient: Gradient(stops: [
                .init(color: Color(platformBackground), location: 0),
                .init(color: Color(platformBackground).opacity(0), location: 1)
            ]),
            startPoint: .top,
            endPoint: .bottom
        )
        .frame(height: topFadeHeight)
        .frame(maxWidth: .infinity, alignment: .top)
        .allowsHitTesting(false)
    }
    #endif

    private final class StreamingBuffer {
        var latest: String = ""
        var isScheduled = false
    }
    
    private final class GeometryBuffer {
        var latest: Bool?
        var isScheduled = false
    }

    private final class ChevronBuffer {
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

    // Use cached messages to avoid recomputing on every layout pass
    private var visibleMessages: [ChatMessage] {
        cachedVisibleMessages
    }

    private func updateCachedVisibleMessages() {
        var messages = chatVM.currentConversation?.messages.filter { $0.role != .system } ?? []
        // Avoid duplicating the streaming assistant message: we render it separately while generating.
        if chatVM.isGenerating, let last = messages.last, last.role == .assistant, !last.isComplete {
            messages.removeLast()
        }
        // Only update if actually changed to avoid triggering unnecessary redraws
        if messages.map(\.id) != cachedVisibleMessages.map(\.id) {
            cachedVisibleMessages = messages
        }
    }

    private var typingIndicator: some View {
        FlameDots(isActive: chatVM.isGenerating, size: 6, spacing: 4)
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

    private func markScrollMovement(durationMs: Int? = nil, recordActivity: Bool = true) {
        if recordActivity {
            recordScrollActivity()
        }
        isScrollMovementActive = true
        scrollMovementTask?.cancel()
        let delay = durationMs ?? scrollMovementIdleMs
        scrollMovementTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(delay))
            isScrollMovementActive = false
            // Don't call updateChevronVisibility here - wait for suspension window
        }
    }

    private func pauseAutoFollowDueToUserScroll() {
        autoFollowSuspendedByUser = true
        if !isScrollMovementActive {
            setChevronVisibility(true, reason: "pause-auto-follow")
        }
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
        beginProgrammaticScroll(durationMs: 200)
    }

    private func beginProgrammaticScroll(durationMs: Int) {
        isProgrammaticScroll = true
        markScrollMovement(durationMs: durationMs)
        programmaticScrollTask?.cancel()
        programmaticScrollTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(durationMs))
            isProgrammaticScroll = false
        }
    }

    private func isUserDrivenScroll() -> Bool {
        let recent = Date().timeIntervalSince(lastUserScrollTime) < userScrollDetectionWindow
        if !recent && isUserDragging {
            isUserDragging = false
        }
        return recent
    }

    private func clearUserScrollIntent() {
        lastUserScrollTime = .distantPast
        isUserDragging = false
    }

    private func forceScrollToLatestUserMessageTop() {
        Task { @MainActor in
            clearUserScrollIntent()

            // Phase 1: ensure layout catches up at the bottom.
            forceScroll(to: "bottom", anchor: .bottom, durationMs: 600, nudges: 2)

            try? await Task.sleep(for: .milliseconds(sendScrollSettleDelayMs))
            clearUserScrollIntent()

            // Phase 2: move the latest user message to top of viewport.
            if let lastUserMessage = visibleMessages.last(where: { $0.role == .user }) {
                forceScroll(to: lastUserMessage.id, anchor: .top, durationMs: 800, nudges: 3)
            } else {
                forceScroll(to: "bottom", anchor: .bottom, durationMs: 600, nudges: 1)
            }
        }
    }

    private func scrollToLatestUserMessageTop(animated: Bool) {
        guard let lastUserMessage = visibleMessages.last(where: { $0.role == .user }) else { return }

        if animated {
            withAnimation(.easeOut(duration: 0.2)) {
                beginProgrammaticScroll()
                scrollProxy?.scrollTo(lastUserMessage.id, anchor: .top)
            }
        } else {
            beginProgrammaticScroll()
            scrollProxy?.scrollTo(lastUserMessage.id, anchor: .top)
        }
    }

    private func forceScrollToBottom() {
        forceScroll(to: "bottom", anchor: .bottom, durationMs: 800, nudges: 3)
    }

    private func forceScroll(
        to targetId: AnyHashable,
        anchor: UnitPoint,
        durationMs: Int,
        nudges: Int
    ) {
        clearUserScrollIntent()
        beginProgrammaticScroll(durationMs: durationMs)
        scrollProxy?.scrollTo(targetId, anchor: anchor)

        Task { @MainActor in
            // A few follow-up nudges to ensure we land at absolute bottom.
            for _ in 0..<nudges {
                try? await Task.sleep(for: .milliseconds(140))
                if isUserDrivenScroll() {
                    return
                }
                beginProgrammaticScroll(durationMs: max(300, durationMs / 2))
                scrollProxy?.scrollTo(targetId, anchor: anchor)
            }
        }
    }

    private func updateChevronVisibility() {
        // Complete no-op during scroll suspension to prevent any layout work
        guard !isInScrollSuspensionWindow() else { return }

        let hasContent = visibleMessages.count >= 1 || chatVM.isGenerating
        let desired: Bool

        // Simple logic: show chevron if user has scrolled away (suspended auto-follow)
        // No longer using hasContentBelow since we removed onScrollGeometryChange
        if autoFollowSuspendedByUser && hasContent {
            desired = true
        } else if scrollMode == .follow {
            desired = false
        } else if chatVM.isGenerating && userInteractedDuringGeneration {
            desired = true
        } else {
            desired = false
        }

        setChevronVisibility(desired, reason: "update")
    }

    private func setChevronVisibility(_ desired: Bool, reason: String) {
        guard showScrollToBottom != desired else { return }

        if shouldDeferChevronUpdate() {
            deferChevronUpdate(desired, reason: reason)
            return
        }

        applyChevronVisibility(desired, reason: reason)
    }

    private func shouldDeferChevronUpdate() -> Bool {
        Date().timeIntervalSince(lastScrollActivityTime) < chevronSuspendWindow
    }

    private func deferChevronUpdate(_ desired: Bool, reason: String) {
        chevronBuffer.latest = desired

        if uiTraceEnabled {
            logDebug("[Chevron] defer show=\(desired) reason=\(reason)", category: .ui)
        }

        guard !chevronBuffer.isScheduled else { return }
        chevronBuffer.isScheduled = true

        Task { @MainActor in
            while true {
                let elapsed = Date().timeIntervalSince(lastScrollActivityTime)
                let remaining = chevronSuspendWindow - elapsed
                if remaining <= 0 {
                    break
                }
                let sleepSeconds = min(0.5, remaining)
                let sleepMs = max(50, Int(sleepSeconds * 1000))
                try? await Task.sleep(for: .milliseconds(sleepMs))
            }

            chevronBuffer.isScheduled = false
            guard let pending = chevronBuffer.latest else { return }
            chevronBuffer.latest = nil
            applyChevronVisibility(pending, reason: "deferred:\(reason)")
        }
    }

    private func applyChevronVisibility(_ desired: Bool, reason: String) {
        guard showScrollToBottom != desired else { return }
        if uiTraceEnabled {
            logDebug("[Chevron] show=\(desired) hasBelow=\(hasContentBelow) generating=\(chatVM.isGenerating) reason=\(reason)", category: .ui)
        }
        showScrollToBottom = desired
    }

    private func recordScrollActivity() {
        lastScrollActivityTime = Date()
    }

    private func isInScrollSuspensionWindow() -> Bool {
        Date().timeIntervalSince(lastScrollActivityTime) < chevronSuspendWindow
    }

    private func schedulePostScrollChevronUpdate() {
        // Cancel any existing scheduled update
        chevronBuffer.isScheduled = true
        Task { @MainActor in
            // Wait for suspension window to expire
            try? await Task.sleep(for: .milliseconds(Int(chevronSuspendWindow * 1000) + 100))
            chevronBuffer.isScheduled = false
            updateChevronVisibility()
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
        // Simplified: no scroll operations during streaming to prevent layout feedback loops
        // User can manually scroll and use chevron to return to bottom
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
            guard !Task.isCancelled else { return }
            guard scrollMode == .follow else { return }
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
    let allowSelection: Bool
    let enableMarkup: Bool
    @State private var cursorVisible = true

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            RoundedRectangle(cornerRadius: 2, style: .continuous)
                .fill(gaugeAccent)
                .frame(width: 3)

            VStack(alignment: .leading, spacing: 0) {
                if content.isEmpty {
                    // Show thinking indicator when no content yet
                    thinkingDots
                } else {
                    // Show streaming text with markdown rendering
                    HStack(alignment: .bottom, spacing: 0) {
                        if enableMarkup {
                            MarkdownView(content: content, isUserMessage: false, allowSelection: allowSelection, isMessageComplete: false)
                        } else {
                            Text(content)
                                .selectable(allowSelection)
                                .lineSpacing(3)
                        }

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
        FlameDots(isActive: true, size: 6, spacing: 4)
    }
}

// Flame-style animated dots for streaming/preview state
private struct FlameDots: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var didAppear = false
    @State private var isAnimating = false

    let isActive: Bool
    let size: CGFloat
    let spacing: CGFloat

    private var baseColor: Color {
        Color(red: 1.0, green: 0.62, blue: 0.2)
    }

    private var highlightGradient: RadialGradient {
        RadialGradient(
            gradient: Gradient(colors: [
                Color.white.opacity(0.8),
                Color.white.opacity(0.0)
            ]),
            center: .topLeading,
            startRadius: 0,
            endRadius: size * 0.9
        )
    }

    var body: some View {
        HStack(spacing: spacing) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(baseColor)
                    .overlay(Circle().fill(highlightGradient))
                    .overlay(
                        Circle()
                            .stroke(baseColor.opacity(dotHaloOpacity(isAnimating: isAnimating)), lineWidth: size * 0.35)
                            .scaleEffect(dotHaloScale(isAnimating: isAnimating))
                            .blur(radius: size * 0.35)
                    )
                    .frame(width: size, height: size)
                    .scaleEffect(dotScale(isAnimating: isAnimating))
                    .opacity(dotOpacity(isAnimating: isAnimating))
                    .offset(y: dotOffset(isAnimating: isAnimating))
                    .shadow(color: Color(red: 1.0, green: 0.55, blue: 0.25).opacity(0.35), radius: 3, x: 0, y: 0)
                    .animation(dotAnimation(for: index), value: isAnimating)
            }
        }
        .onAppear {
            guard !didAppear else { return }
            didAppear = true
            // Defer animation start to next run loop so initial layout is settled
            DispatchQueue.main.async {
                updateAnimation()
            }
        }
        .onChange(of: isActive) { _, _ in
            updateAnimation()
        }
    }

    // Use midpoint values as the "resting" state so the initial appearance doesn't jump
    private func dotScale(isAnimating: Bool) -> CGFloat {
        if reduceMotion || !isActive {
            return 0.85
        }
        return isAnimating ? 1.0 : 0.8
    }

    private func dotOpacity(isAnimating: Bool) -> Double {
        if reduceMotion || !isActive {
            return 0.55
        }
        return isAnimating ? 1.0 : 0.5
    }

    private func dotOffset(isAnimating: Bool) -> CGFloat {
        if reduceMotion || !isActive {
            return 0
        }
        return isAnimating ? -size * 0.15 : size * 0.08
    }

    private func dotHaloOpacity(isAnimating: Bool) -> Double {
        if reduceMotion || !isActive {
            return 0.0
        }
        return isAnimating ? 0.35 : 0.0
    }

    private func dotHaloScale(isAnimating: Bool) -> CGFloat {
        if reduceMotion || !isActive {
            return 1.0
        }
        return isAnimating ? 1.4 : 1.1
    }

    private func dotAnimation(for index: Int) -> Animation {
        guard !reduceMotion, isActive else {
            return .default
        }
        return .easeInOut(duration: 0.55)
            .repeatForever(autoreverses: true)
            .delay(Double(index) * 0.16)
    }

    private func updateAnimation() {
        guard !reduceMotion, isActive else {
            isAnimating = false
            return
        }
        isAnimating = true
    }
}

// MARK: - Model Loading Gauge

struct ModelLoadingGauge: View {
    let progress: ModelLoadingProgress
    var modelName: String? = nil

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var startTime: Date?

    private var clampedProgress: Double {
        min(max(progress.percentage, 0), 1)
    }

    private var barWidth: CGFloat {
        #if os(macOS)
        return 280
        #else
        // Use a more adaptive width for iPhone - not too wide
        return 240
        #endif
    }

    private var barHeight: CGFloat {
        #if os(macOS)
        return 14
        #else
        return 16
        #endif
    }

    private var barCornerRadius: CGFloat {
        barHeight / 2
    }

    private var stripesOverlay: some View {
        Group {
            if reduceMotion {
                DiagonalStripes(
                    color: gaugeAccent.opacity(0.45),
                    lineWidth: 2,
                    spacing: 7,
                    angle: .degrees(60),
                    phase: 0
                )
            } else {
                TimelineView(.animation(minimumInterval: 0.05, paused: false)) { context in
                    DiagonalStripes(
                        color: gaugeAccent.opacity(0.45),
                        lineWidth: 2,
                        spacing: 7,
                        angle: .degrees(60),
                        phase: CGFloat(context.date.timeIntervalSinceReferenceDate * 26)
                    )
                }
            }
        }
    }

    private func estimatedSecondsRemaining(now: Date) -> Int? {
        guard let start = startTime,
              clampedProgress > 0.05 else { return nil } // Need at least 5% to estimate

        let elapsed = max(0.1, now.timeIntervalSince(start))
        let progressRate = clampedProgress / elapsed
        guard progressRate > 0 else { return nil }

        let remaining = (1.0 - clampedProgress) / progressRate
        return max(1, Int(ceil(remaining)))
    }

    private func etaString(now: Date) -> String? {
        guard let seconds = estimatedSecondsRemaining(now: now) else { return nil }
        if seconds < 60 {
            return "\(seconds)s"
        } else {
            let minutes = seconds / 60
            let secs = seconds % 60
            return "\(minutes)m \(secs)s"
        }
    }

    var body: some View {
        VStack(spacing: 10) {
            VStack(spacing: 2) {
                Text(progress.stage)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                // Only show detail if it's not a file path (hide technical paths from users)
                if let detail = progress.detail, !detail.isEmpty, !detail.contains("/") {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                }
            }

            VStack(spacing: 8) {
                ZStack {
                    RoundedRectangle(cornerRadius: barCornerRadius, style: .continuous)
                        .fill(Color.black.opacity(0.35))
                        .overlay(
                            RoundedRectangle(cornerRadius: barCornerRadius, style: .continuous)
                                .stroke(gaugeAccent.opacity(0.45), lineWidth: 1)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: barCornerRadius, style: .continuous)
                                .stroke(
                                    LinearGradient(
                                        colors: [
                                            Color.white.opacity(0.35),
                                            Color.clear
                                        ],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    ),
                                    lineWidth: 0.8
                                )
                        )

                    GeometryReader { geometry in
                        let width = max(0, geometry.size.width * clampedProgress)

                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: barCornerRadius, style: .continuous)
                                .fill(
                                    LinearGradient(
                                        colors: [
                                            gaugeAccent.opacity(0.25),
                                            gaugeAccent,
                                            Color.white.opacity(0.9)
                                        ],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: width)
                                .shadow(color: gaugeAccent.opacity(0.45), radius: 8, x: 0, y: 0)

                            if width > 0.5 {
                                stripesOverlay
                                    .frame(width: width, height: geometry.size.height)
                                    .mask(
                                        RoundedRectangle(cornerRadius: barCornerRadius, style: .continuous)
                                            .frame(width: width, height: geometry.size.height)
                                    )
                            }

                            RoundedRectangle(cornerRadius: barCornerRadius, style: .continuous)
                                .stroke(
                                    LinearGradient(
                                        colors: [
                                            Color.white.opacity(0.6),
                                            Color.clear
                                        ],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    ),
                                    lineWidth: 0.8
                                )
                                .frame(width: width)
                        }
                    }
                }
                .frame(width: barWidth, height: barHeight)

                HStack(spacing: 10) {
                    Text("Loading...")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(gaugeAccent.opacity(0.85))
                        .textCase(.uppercase)

                    Text("\(Int(clampedProgress * 100))%")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(.primary)
                        .monospacedDigit()

                    TimelineView(.periodic(from: .now, by: 1)) { context in
                        if let eta = etaString(now: context.date) {
                            Text(eta)
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                    }
                }

                // Model name display
                if let name = modelName, !name.isEmpty {
                    Text(name)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
            }
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 18)
        .background(modelLoadingBackground, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        )
        .onAppear {
            if startTime == nil {
                startTime = Date()
            }
        }
        .onChange(of: clampedProgress) { oldValue, newValue in
            // Reset timer if progress restarts
            if newValue < oldValue - 0.1 {
                startTime = Date()
            }
        }
    }
}

// MARK: - Diagonal Stripes

private struct DiagonalStripes: View {
    let color: Color
    let lineWidth: CGFloat
    let spacing: CGFloat
    let angle: Angle
    let phase: CGFloat

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                let diagonal = hypot(size.width, size.height)
                let step = max(1, lineWidth + spacing)
                let count = Int(diagonal / step) + 3
                let offset = phase.truncatingRemainder(dividingBy: step)

                var path = Path()
                for index in 0..<count {
                    let x = CGFloat(index) * step - offset
                    path.move(to: CGPoint(x: x, y: -diagonal))
                    path.addLine(to: CGPoint(x: x + diagonal, y: diagonal))
                }

                context.stroke(path, with: .color(color), lineWidth: lineWidth)
            }
            .rotationEffect(angle)
        }
    }
}

// MARK: - Platform Colors & Helpers

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
private let gaugeAccent = Color(red: 1.0, green: 0.62, blue: 0.2)

private enum UIScreen {
    static var mainHeight: CGFloat {
        UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first?.screen.bounds.height ?? 800
    }
}
#else
private let platformBackground = NSColor.windowBackgroundColor
private let chatBackground = Color(platformBackground)
private let platformSecondaryBackground = NSColor.controlBackgroundColor
private let modelLoadingBackground = Color(platformSecondaryBackground)
private let gaugeAccent = Color(red: 1.0, green: 0.62, blue: 0.2)

private enum UIScreen {
    static var mainHeight: CGFloat {
        NSScreen.main?.frame.height ?? 800
    }
}
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

#if os(macOS)
private struct PointingHandCursorModifier: ViewModifier {
    func body(content: Content) -> some View {
        content.onHover { hovering in
            if hovering {
                NSCursor.pointingHand.set()
            } else {
                NSCursor.arrow.set()
            }
        }
    }
}

private extension View {
    func pointingHandCursor() -> some View {
        modifier(PointingHandCursorModifier())
    }
}
#endif

#Preview {
    NavigationStack {
        ChatView()
            .environment(ChatViewModel())
            .environment(ModelManagerViewModel())
    }
}
