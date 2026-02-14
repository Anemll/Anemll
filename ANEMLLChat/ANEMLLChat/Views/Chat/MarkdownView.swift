//
//  MarkdownView.swift
//  ANEMLLChat
//
//  Renders markdown content with support for tables, lists, code blocks
//

import SwiftUI
#if os(macOS)
import AppKit
#else
import UIKit
#endif

struct MarkdownView: View {
    let content: String
    let isUserMessage: Bool
    let allowSelection: Bool
    var isMessageComplete: Bool = true  // Whether the parent message is complete

    @State private var cachedContent: String = ""
    @State private var cachedBlocks: [MarkdownBlock] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(cachedBlocks.enumerated()), id: \.offset) { _, block in
                renderBlock(block)
            }
        }
        .onAppear {
            refreshCacheIfNeeded()
        }
        .onChange(of: content) { _, _ in
            refreshCacheIfNeeded()
        }
    }

    // MARK: - Block Types

    private enum MarkdownBlock {
        case paragraph(String)
        case codeBlock(String, language: String?)
        case table(headers: [String], rows: [[String]])
        case bulletList([String])
        case numberedList([String])
        case heading(String, level: Int)
        case thinkBlock(String)  // <think>...</think> collapsible block
    }

    // MARK: - Parsing

    private func parseBlocks(from content: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []

        // First, extract <think>...</think> blocks
        let processedContent = extractThinkBlocks(from: content, into: &blocks)

        let lines = processedContent.components(separatedBy: "\n")
        var i = 0

        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Code block
            if trimmed.hasPrefix("```") {
                let language = String(trimmed.dropFirst(3))
                var codeLines: [String] = []
                i += 1
                while i < lines.count && !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    codeLines.append(lines[i])
                    i += 1
                }
                blocks.append(.codeBlock(codeLines.joined(separator: "\n"), language: language.isEmpty ? nil : language))
                i += 1
                continue
            }

            // Table (starts with |)
            if trimmed.hasPrefix("|") && trimmed.contains("|") {
                var tableLines: [String] = []
                while i < lines.count && lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("|") {
                    tableLines.append(lines[i])
                    i += 1
                }
                if let table = parseTable(tableLines) {
                    blocks.append(table)
                }
                continue
            }

            // Heading
            if trimmed.hasPrefix("#") {
                let level = trimmed.prefix(while: { $0 == "#" }).count
                let text = String(trimmed.dropFirst(level)).trimmingCharacters(in: .whitespaces)
                blocks.append(.heading(text, level: min(level, 6)))
                i += 1
                continue
            }

            // Bullet list
            if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") || trimmed.hasPrefix("• ") {
                var items: [String] = []
                while i < lines.count {
                    let t = lines[i].trimmingCharacters(in: .whitespaces)
                    if t.hasPrefix("- ") || t.hasPrefix("* ") || t.hasPrefix("• ") {
                        items.append(String(t.dropFirst(2)))
                        i += 1
                    } else {
                        break
                    }
                }
                blocks.append(.bulletList(items))
                continue
            }

            // Numbered list
            if let _ = trimmed.range(of: #"^\d+\.\s"#, options: .regularExpression) {
                var items: [String] = []
                while i < lines.count {
                    let t = lines[i].trimmingCharacters(in: .whitespaces)
                    if let range = t.range(of: #"^\d+\.\s"#, options: .regularExpression) {
                        items.append(String(t[range.upperBound...]))
                        i += 1
                    } else {
                        break
                    }
                }
                blocks.append(.numberedList(items))
                continue
            }

            // Empty line - skip
            if trimmed.isEmpty {
                i += 1
                continue
            }

            // Skip orphaned markdown markers (e.g., just "**" on a line)
            if trimmed == "**" || trimmed == "*" || trimmed == "__" || trimmed == "_" {
                i += 1
                continue
            }

            // Paragraph - collect consecutive non-special lines
            var paragraphLines: [String] = []
            while i < lines.count {
                let t = lines[i].trimmingCharacters(in: .whitespaces)
                if t.isEmpty || t.hasPrefix("#") || t.hasPrefix("|") || t.hasPrefix("```") ||
                   t.hasPrefix("- ") || t.hasPrefix("* ") || t.hasPrefix("• ") ||
                   t.range(of: #"^\d+\.\s"#, options: .regularExpression) != nil {
                    break
                }
                paragraphLines.append(lines[i])
                i += 1
            }
            if !paragraphLines.isEmpty {
                blocks.append(.paragraph(paragraphLines.joined(separator: " ")))
            }
        }

        return blocks
    }

    /// Extract <think>...</think> blocks and return content with placeholders replaced
    private func extractThinkBlocks(from content: String, into blocks: inout [MarkdownBlock]) -> String {
        var result = content

        // First, handle complete <think>...</think> blocks
        let completePattern = #"<think>([\s\S]*?)</think>"#
        if let regex = try? NSRegularExpression(pattern: completePattern, options: []) {
            let range = NSRange(content.startIndex..., in: content)
            let matches = regex.matches(in: content, options: [], range: range)

            // Process matches in reverse order to preserve indices
            for match in matches.reversed() {
                guard let thinkRange = Range(match.range(at: 1), in: content) else { continue }
                let thinkContent = String(content[thinkRange]).trimmingCharacters(in: .whitespacesAndNewlines)

                // Add think block
                blocks.insert(.thinkBlock(thinkContent), at: 0)

                // Remove the <think>...</think> from result
                if let fullRange = Range(match.range, in: result) {
                    result.replaceSubrange(fullRange, with: "")
                }
            }
        }

        // Handle incomplete <think> without closing </think> (streaming response)
        if let openRange = result.range(of: "<think>") {
            // There's an unclosed <think> tag - extract content and show as in-progress thinking
            let thinkContent = String(result[openRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !thinkContent.isEmpty {
                // Use special marker that won't appear in natural text
                blocks.append(.thinkBlock(thinkContent + " \u{2026}"))  // Unicode ellipsis character
            } else {
                blocks.append(.thinkBlock("\u{2026}"))  // Just started thinking
            }
            result = String(result[..<openRange.lowerBound])
        }

        return result
    }

    private func refreshCacheIfNeeded() {
        guard content != cachedContent else { return }
        cachedContent = content
        cachedBlocks = parseBlocks(from: content)
    }

    private func parseTable(_ lines: [String]) -> MarkdownBlock? {
        guard lines.count >= 2 else { return nil }

        func parseRow(_ line: String) -> [String] {
            line.split(separator: "|", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
        }

        let headers = parseRow(lines[0])

        // Skip separator line (|---|---|)
        let dataStartIndex = lines.count > 1 && lines[1].contains("-") ? 2 : 1

        var rows: [[String]] = []
        for i in dataStartIndex..<lines.count {
            let row = parseRow(lines[i])
            if !row.isEmpty {
                rows.append(row)
            }
        }

        return .table(headers: headers, rows: rows)
    }

    // MARK: - Rendering

    @ViewBuilder
    private func renderBlock(_ block: MarkdownBlock) -> some View {
        switch block {
        case .paragraph(let text):
            renderInlineMarkdown(text)

        case .codeBlock(let code, let language):
            CodeBlockView(code: code, language: language)

        case .table(let headers, let rows):
            renderTable(headers: headers, rows: rows)

        case .bulletList(let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .top, spacing: 8) {
                        Text("•")
                        renderInlineMarkdown(item)
                    }
                }
            }

        case .numberedList(let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                    HStack(alignment: .top, spacing: 8) {
                        Text("\(index + 1).")
                            .frame(minWidth: 20, alignment: .trailing)
                        renderInlineMarkdown(item)
                    }
                }
            }

        case .heading(let text, let level):
            renderInlineMarkdown(text)
                .font(headingFont(level: level))
                .fontWeight(.bold)

        case .thinkBlock(let thinkContent):
            ThinkBlockView(content: thinkContent, allowSelection: allowSelection, isMessageComplete: isMessageComplete)
        }
    }

    private func headingFont(level: Int) -> Font {
        switch level {
        case 1: return .title
        case 2: return .title2
        case 3: return .title3
        default: return .headline
        }
    }

    @ViewBuilder
    private func renderTable(headers: [String], rows: [[String]]) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header row
            HStack(spacing: 0) {
                ForEach(Array(headers.enumerated()), id: \.offset) { index, header in
                    renderInlineMarkdown(header)
                        .font(.system(.caption, design: .monospaced))
                        .fontWeight(.bold)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if index < headers.count - 1 {
                        Divider()
                    }
                }
            }
            .background(Color.secondary.opacity(0.2))

            Divider()

            // Data rows
            ForEach(Array(rows.enumerated()), id: \.offset) { rowIndex, row in
                HStack(spacing: 0) {
                    ForEach(Array(row.enumerated()), id: \.offset) { colIndex, cell in
                        renderInlineMarkdown(cell)
                            .font(.system(.caption, design: .monospaced))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        if colIndex < row.count - 1 {
                            Divider()
                        }
                    }
                }
                if rowIndex < rows.count - 1 {
                    Divider()
                }
            }
        }
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
        )
    }

    @ViewBuilder
    private func renderInlineMarkdown(_ text: String) -> some View {
        InlineMarkdownText(text: text, allowSelection: allowSelection)
    }
}

private struct InlineMarkdownText: View {
    let text: String
    let allowSelection: Bool

    @State private var cachedText: String = ""
    @State private var cachedAttributed: AttributedString?

    var body: some View {
        Group {
            if let attributed = cachedAttributed {
                Text(attributed)
            } else {
                Text(text)
            }
        }
        .selectable(allowSelection)
        .lineSpacing(3)
        .onAppear {
            refreshCacheIfNeeded()
        }
        .onChange(of: text) { _, _ in
            refreshCacheIfNeeded()
        }
    }

    private func refreshCacheIfNeeded() {
        guard text != cachedText else { return }
        cachedText = text
        if let attributed = try? AttributedString(
            markdown: text,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            cachedAttributed = attributed
        } else {
            cachedAttributed = nil
        }
    }
}

// MARK: - Think Block View (Collapsible)

/// Collapsible view for <think>...</think> content
struct ThinkBlockView: View {
    let content: String
    let allowSelection: Bool
    var isMessageComplete: Bool = false  // Whether the parent message generation is complete

    @State private var isExpanded: Bool = false
    @State private var thinkingStartTime: Date?
    @State private var thinkingDuration: TimeInterval = 0

    // Orange shade - similar to LLM accent but slightly different
    private var thinkAccent: Color {
        Color(red: 0.95, green: 0.55, blue: 0.2)  // Warm orange thinking color
    }

    /// Check if thinking is still in progress
    /// Thinking is complete if:
    /// 1. The parent message is complete (generation finished), OR
    /// 2. The content doesn't have the ellipsis marker (proper </think> was found)
    private var isThinking: Bool {
        // If message is complete, thinking is definitely done
        if isMessageComplete {
            return false
        }
        // Otherwise check for the streaming marker
        return content.hasSuffix("\u{2026}")  // Unicode ellipsis character
    }

    /// Computed duration - uses recorded duration or estimates if needed
    private var displayDuration: TimeInterval {
        if thinkingDuration > 0 {
            return thinkingDuration
        }
        // Fallback: estimate based on content length
        // ~20 tokens per second, ~4 chars per token
        let estimatedTokens = max(1, content.count / 4)
        return max(1, Double(estimatedTokens) / 20.0)
    }

    /// Format duration for display
    private var durationText: String {
        let seconds = Int(displayDuration)
        if seconds < 60 {
            return "\(seconds)s"
        } else {
            let mins = seconds / 60
            let secs = seconds % 60
            return "\(mins):\(String(format: "%02d", secs))"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header - always visible, tappable
            Button {
                withAnimation(.easeInOut(duration: 0.25)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 8) {
                    // Think icon
                    Image(systemName: "brain")
                        .font(.caption)
                        .foregroundStyle(thinkAccent)

                    if isThinking {
                        // Still thinking - show "Thinking" with animated dots
                        Text("Thinking")
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundStyle(thinkAccent)

                        // Animated dots when collapsed and still thinking
                        if !isExpanded {
                            ThinkingDotsView(color: thinkAccent)
                        }
                    } else {
                        // Done thinking - show "Thought for X"
                        Text("Thought for \(durationText)")
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundStyle(thinkAccent)
                    }

                    Spacer()

                    // Expand/collapse chevron
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(thinkAccent.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }
            .buttonStyle(.plain)

            // Content - shown when expanded
            if isExpanded {
                VStack(alignment: .leading, spacing: 4) {
                    Divider()
                        .background(thinkAccent.opacity(0.3))

                    // Remove trailing ellipsis marker from display content if present
                    let displayContent = content.hasSuffix(" \u{2026}")
                        ? String(content.dropLast(2))  // Remove " …"
                        : (content.hasSuffix("\u{2026}") ? String(content.dropLast(1)) : content)

                    Text(displayContent)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .selectable(allowSelection)
                        .lineSpacing(3)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                }
                .background(thinkAccent.opacity(0.05))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(thinkAccent.opacity(0.2), lineWidth: 1)
        )
        .onAppear {
            initializeThinkingState()
        }
        .onChange(of: content) { _, newContent in
            // When content changes and thinking completes, record duration
            if !newContent.hasSuffix("\u{2026}") && thinkingStartTime != nil && thinkingDuration == 0 {
                thinkingDuration = Date().timeIntervalSince(thinkingStartTime!)
            }
        }
        .onChange(of: isThinking) { wasThinking, nowThinking in
            // Thinking just finished
            if wasThinking && !nowThinking {
                if let start = thinkingStartTime, thinkingDuration == 0 {
                    thinkingDuration = Date().timeIntervalSince(start)
                }
            }
        }
    }

    /// Initialize thinking state on appear
    private func initializeThinkingState() {
        if isThinking {
            // Start tracking thinking time
            if thinkingStartTime == nil {
                thinkingStartTime = Date()
            }
        }
        // Note: displayDuration computed property handles fallback estimation
    }
}

/// Self-contained animated dots view that manages its own animation state
private struct ThinkingDotsView: View {
    let color: Color

    @State private var isAnimating = false

    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(color)
                    .frame(width: 5, height: 5)
                    .scaleEffect(isAnimating ? 1.4 : 1.0)
                    .animation(
                        .easeInOut(duration: 0.4)
                        .repeatForever(autoreverses: true)
                        .delay(Double(index) * 0.15),
                        value: isAnimating
                    )
            }
        }
        .onAppear {
            // Small delay to ensure view is fully rendered before animating
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                isAnimating = true
            }
        }
    }
}

// MARK: - Code Block View (with Copy Button)

/// Code block with hover-to-copy functionality
private struct CodeBlockView: View {
    let code: String
    let language: String?

    @State private var isHovering = false
    @State private var showCopied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            // Language label and copy button row
            HStack {
                if let lang = language, !lang.isEmpty {
                    Text(lang)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                // Copy button (visible on hover)
                Button {
                    copyCode()
                } label: {
                    HStack(spacing: 3) {
                        Image(systemName: showCopied ? "checkmark" : "doc.on.doc")
                            .font(.caption2)
                        if showCopied {
                            Text("Copied")
                                .font(.caption2)
                        }
                    }
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 4))
                }
                .buttonStyle(.plain)
                .opacity(isHovering || showCopied ? 1 : 0)
            }

            // Code content
            Text(code)
                .font(.system(.body, design: .monospaced))
                .textSelection(.enabled)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.black.opacity(0.3))
                .cornerRadius(8)
        }
        #if os(macOS)
        .onHover { hovering in
            isHovering = hovering
            if !hovering {
                // Reset copied state when mouse leaves
                showCopied = false
            }
        }
        #endif
    }

    private func copyCode() {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)
        #else
        UIPasteboard.general.string = code
        #endif

        // Show "Copied" feedback
        withAnimation(.easeInOut(duration: 0.2)) {
            showCopied = true
        }

        // Reset after delay
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            await MainActor.run {
                withAnimation(.easeInOut(duration: 0.2)) {
                    showCopied = false
                }
            }
        }
    }
}

#Preview("Think Block") {
    VStack(spacing: 20) {
        // In-progress thinking (streaming - uses ellipsis marker)
        ThinkBlockView(
            content: "Let me think about this step by step \u{2026}",
            allowSelection: true,
            isMessageComplete: false
        )

        // Completed thinking (message complete, even with ellipsis)
        ThinkBlockView(
            content: "I analyzed the problem and determined the solution...\n1. First step\n2. Second step\n3. Third step",
            allowSelection: true,
            isMessageComplete: true
        )
    }
    .padding()
}

#Preview {
    ScrollView {
        VStack(alignment: .leading, spacing: 20) {
            MarkdownView(content: """
            # Heading 1
            ## Heading 2

            This is a paragraph with **bold** and *italic* text.

            <think>
            Let me think about this...
            I need to consider multiple factors.
            </think>

            - First item
            - Second item
            - Third item

            1. Numbered one
            2. Numbered two
            3. Numbered three

            | Feature | TPU | ANE |
            |---------|-----|-----|
            | Cost | High | Low |
            | Speed | Fast | Fast |

            ```swift
            let x = 10
            print(x)
            ```
            """, isUserMessage: false, allowSelection: true)
        }
        .padding()
    }
}
