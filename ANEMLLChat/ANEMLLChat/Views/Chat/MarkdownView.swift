//
//  MarkdownView.swift
//  ANEMLLChat
//
//  Renders markdown content with support for tables, lists, code blocks
//

import SwiftUI

struct MarkdownView: View {
    let content: String
    let isUserMessage: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(parseBlocks().enumerated()), id: \.offset) { _, block in
                renderBlock(block)
            }
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
    }

    // MARK: - Parsing

    private func parseBlocks() -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        let lines = content.components(separatedBy: "\n")
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
            VStack(alignment: .leading, spacing: 4) {
                if let lang = language, !lang.isEmpty {
                    Text(lang)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Text(code)
                    .font(.system(.body, design: .monospaced))
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.black.opacity(0.3))
                    .cornerRadius(8)
            }

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
        if let attributed = try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
            Text(attributed)
                .textSelection(.enabled)
                .lineSpacing(3)
        } else {
            Text(text)
                .textSelection(.enabled)
                .lineSpacing(3)
        }
    }
}

#Preview {
    ScrollView {
        VStack(alignment: .leading, spacing: 20) {
            MarkdownView(content: """
            # Heading 1
            ## Heading 2

            This is a paragraph with **bold** and *italic* text.

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
            """, isUserMessage: false)
        }
        .padding()
    }
}
