import Foundation
import Network
import ArgumentParser
import AnemllCore
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

enum CLIServerBindMode: String, Sendable {
    case localhost
    case lan

    init(parsing rawValue: String) throws {
        let normalized = rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let mode = CLIServerBindMode(rawValue: normalized) else {
            throw ValidationError("Invalid --serve-bind value '\(rawValue)'. Use 'localhost' or 'lan'.")
        }
        self = mode
    }
}

actor CLIGenerationGate {
    private var busy = false

    func acquire() -> Bool {
        guard !busy else { return false }
        busy = true
        return true
    }

    func release() {
        busy = false
    }
}

final class CLIOpenAICompatibleServer: @unchecked Sendable {
    private let inferenceManager: InferenceManager
    private let tokenizer: Tokenizer
    private let defaultTemperature: Float
    private let defaultMaxTokens: Int
    private let modelID: String
    private let defaultSystemPrompt: String?
    private let addGenerationPrompt: Bool
    private let generationGate = CLIGenerationGate()

    private let listenerQueue = DispatchQueue(label: "com.anemll.cli.openai-server")
    private var listener: NWListener?

    init(
        inferenceManager: InferenceManager,
        tokenizer: Tokenizer,
        defaultTemperature: Float,
        defaultMaxTokens: Int,
        modelID: String,
        defaultSystemPrompt: String?,
        addGenerationPrompt: Bool
    ) {
        self.inferenceManager = inferenceManager
        self.tokenizer = tokenizer
        self.defaultTemperature = defaultTemperature
        self.defaultMaxTokens = defaultMaxTokens
        self.modelID = modelID
        self.defaultSystemPrompt = defaultSystemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.addGenerationPrompt = addGenerationPrompt
    }

    func start(bindMode: CLIServerBindMode, port: Int) throws {
        stop()

        let normalizedPort = Self.normalizedPort(port)
        guard let nwPort = NWEndpoint.Port(rawValue: UInt16(normalizedPort)) else {
            throw ValidationError("Invalid --serve-port value '\(port)'.")
        }

        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true

        let listener: NWListener
        if bindMode == .localhost {
            params.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: nwPort)
            listener = try NWListener(using: params)
        } else {
            listener = try NWListener(using: params, on: nwPort)
        }

        listener.stateUpdateHandler = { state in
            switch state {
            case .ready:
                print("OpenAI-compatible server ready on \(bindMode == .localhost ? "localhost" : "lan"):\(normalizedPort)")
            case .failed(let error):
                print("Server failed: \(error.localizedDescription)")
            case .waiting(let error):
                print("Server waiting: \(error.localizedDescription)")
            default:
                break
            }
        }

        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            connection.start(queue: self.listenerQueue)
            Self.receiveRequest(on: connection, accumulated: Data(), mode: bindMode) { request in
                Task {
                    let response = await self.response(for: request)
                    Self.send(response, on: connection)
                }
            }
        }

        self.listener = listener
        listener.start(queue: listenerQueue)

        print("Local URL: http://127.0.0.1:\(normalizedPort)")
        if bindMode == .lan, let lanAddress = Self.preferredLANIPv4Address() {
            print("LAN URL:   http://\(lanAddress):\(normalizedPort)")
        }
        print("Endpoints: GET /v1/models, POST /v1/chat/completions")
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    private func response(for request: HTTPRequest) async -> HTTPResponse {
        let path = Self.normalizePath(request.path)
        switch (request.method, path) {
        case ("GET", "/v1/models"):
            let payload = ModelsListResponse(
                object: "list",
                data: [ModelDescriptor(
                    id: modelID,
                    object: "model",
                    created: Int(Date().timeIntervalSince1970),
                    ownedBy: "anemll"
                )]
            )
            return Self.json(statusCode: 200, payload: payload)

        case ("POST", "/v1/chat/completions"):
            return await handleChatCompletionsRequest(body: request.body)

        case ("GET", "/health"):
            return Self.json(statusCode: 200, payload: ["status": "ok"])

        default:
            return Self.errorResponse(
                statusCode: 404,
                message: "Path not found: \(path)",
                code: "not_found"
            )
        }
    }

    private func handleChatCompletionsRequest(body: Data) async -> HTTPResponse {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        let request: ChatCompletionsRequest
        do {
            request = try decoder.decode(ChatCompletionsRequest.self, from: body)
        } catch {
            return Self.errorResponse(
                statusCode: 400,
                message: "Invalid JSON body: \(error.localizedDescription)",
                code: "invalid_json"
            )
        }

        if request.stream == true {
            return Self.errorResponse(
                statusCode: 400,
                message: "Streaming is not supported yet. Set stream=false.",
                code: "stream_not_supported"
            )
        }

        if let requestedModel = request.model?.trimmingCharacters(in: .whitespacesAndNewlines),
           !requestedModel.isEmpty,
           requestedModel != modelID,
           requestedModel != "anemll-current-model" {
            return Self.errorResponse(
                statusCode: 400,
                message: "Requested model '\(requestedModel)' is not active. Active model: '\(modelID)'.",
                code: "model_mismatch"
            )
        }

        guard await generationGate.acquire() else {
            return Self.errorResponse(
                statusCode: 429,
                message: "Model is currently generating another response.",
                code: "model_busy"
            )
        }
        defer {
            Task { await generationGate.release() }
        }

        var messages = request.messages.compactMap { message -> Tokenizer.ChatMessage? in
            let text = message.flattenedText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }

            switch message.role.lowercased() {
            case "system", "developer":
                return .system(text)
            case "user":
                return .user(text)
            case "assistant":
                return .assistant(text)
            default:
                return nil
            }
        }

        if !messages.contains(where: { $0.role == "system" }),
           let defaultSystemPrompt,
           !defaultSystemPrompt.isEmpty {
            messages.insert(.system(defaultSystemPrompt), at: 0)
        }

        guard !messages.isEmpty else {
            return Self.errorResponse(
                statusCode: 400,
                message: "Request must include at least one valid message.",
                code: "empty_messages"
            )
        }

        let inputTokens = tokenizer.applyChatTemplate(
            input: messages,
            addGenerationPrompt: addGenerationPrompt
        )

        let requestTemperature = request.temperature.map { Float(max(0.0, min(2.0, $0))) } ?? defaultTemperature
        let requestMaxTokens = request.maxCompletionTokens ?? request.maxTokens ?? defaultMaxTokens
        let clampedMaxTokens = max(1, requestMaxTokens)

        do {
            let (generatedTokens, _, stopReason) = try await inferenceManager.generateResponse(
                initialTokens: inputTokens,
                temperature: requestTemperature,
                maxTokens: clampedMaxTokens,
                eosTokens: tokenizer.eosTokenIds,
                tokenizer: tokenizer,
                onToken: { _ in }
            )

            let responseText = tokenizer.decode(tokens: generatedTokens)
            let usage = Usage(
                promptTokens: inputTokens.count,
                completionTokens: generatedTokens.count,
                totalTokens: inputTokens.count + generatedTokens.count
            )

            let response = ChatCompletionsResponse(
                id: "chatcmpl-\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))",
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: modelID,
                choices: [
                    Choice(
                        index: 0,
                        message: ChoiceMessage(role: "assistant", content: responseText),
                        finishReason: Self.finishReason(
                            stopReason: stopReason,
                            completionTokens: generatedTokens.count,
                            maxTokens: clampedMaxTokens
                        )
                    )
                ],
                usage: usage
            )

            return Self.json(statusCode: 200, payload: response)
        } catch {
            return Self.errorResponse(
                statusCode: 500,
                message: "Generation failed: \(error.localizedDescription)",
                code: "generation_failed"
            )
        }
    }
}

// MARK: - HTTP

private extension CLIOpenAICompatibleServer {
    static let headerDelimiter = Data("\r\n\r\n".utf8)
    static let maxRequestBytes = 4_194_304

    struct HTTPRequest {
        let method: String
        let path: String
        let body: Data
    }

    struct HTTPResponse {
        let statusCode: Int
        let reasonPhrase: String
        let headers: [String: String]
        let body: Data
    }

    enum ParseResult {
        case needMoreData
        case malformed(String)
        case complete(HTTPRequest)
    }

    static func receiveRequest(
        on connection: NWConnection,
        accumulated: Data,
        mode: CLIServerBindMode,
        requestHandler: @escaping @Sendable (HTTPRequest) -> Void
    ) {
        if mode == .localhost,
           let remoteHost = remoteHostString(from: connection.endpoint),
           !isLoopbackHost(remoteHost) {
            send(
                errorResponse(
                    statusCode: 403,
                    message: "Connection denied. Localhost mode accepts only loopback clients.",
                    code: "forbidden_remote_host"
                ),
                on: connection
            )
            return
        }

        connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { data, _, isComplete, error in
            var buffer = accumulated
            if let data, !data.isEmpty {
                buffer.append(data)
            }

            if buffer.count > maxRequestBytes {
                send(
                    errorResponse(
                        statusCode: 413,
                        message: "Request body is too large.",
                        code: "request_too_large"
                    ),
                    on: connection
                )
                return
            }

            switch parseRequest(from: buffer) {
            case .complete(let request):
                requestHandler(request)

            case .malformed(let message):
                send(
                    errorResponse(
                        statusCode: 400,
                        message: message,
                        code: "bad_request"
                    ),
                    on: connection
                )

            case .needMoreData:
                if let error {
                    send(
                        errorResponse(
                            statusCode: 400,
                            message: "Receive error: \(error.localizedDescription)",
                            code: "receive_error"
                        ),
                        on: connection
                    )
                    return
                }
                guard !isComplete else {
                    send(
                        errorResponse(
                            statusCode: 400,
                            message: "Connection closed before full request body was received.",
                            code: "incomplete_request"
                        ),
                        on: connection
                    )
                    return
                }
                receiveRequest(on: connection, accumulated: buffer, mode: mode, requestHandler: requestHandler)
            }
        }
    }

    static func parseRequest(from data: Data) -> ParseResult {
        guard let delimiterRange = data.range(of: headerDelimiter) else {
            return .needMoreData
        }

        let headerBytes = data[..<delimiterRange.lowerBound]
        guard let headerString = String(data: headerBytes, encoding: .utf8) else {
            return .malformed("Request headers must be UTF-8.")
        }

        let lines = headerString.components(separatedBy: "\r\n")
        guard let requestLine = lines.first, !requestLine.isEmpty else {
            return .malformed("Missing HTTP request line.")
        }

        let requestParts = requestLine.split(separator: " ")
        guard requestParts.count >= 2 else {
            return .malformed("Malformed HTTP request line: \(requestLine)")
        }

        let method = String(requestParts[0]).uppercased()
        let path = String(requestParts[1])

        var contentLength = 0
        for line in lines.dropFirst() where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = line[..<colon].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if key == "content-length" {
                let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines)
                guard let parsed = Int(value), parsed >= 0 else {
                    return .malformed("Invalid Content-Length header.")
                }
                contentLength = parsed
            }
        }

        let bodyStart = delimiterRange.upperBound
        let expectedTotal = bodyStart + contentLength
        guard data.count >= expectedTotal else {
            return .needMoreData
        }

        let body = Data(data[bodyStart..<expectedTotal])
        return .complete(HTTPRequest(method: method, path: path, body: body))
    }

    static func send(_ response: HTTPResponse, on connection: NWConnection) {
        var headers = response.headers
        headers["Content-Length"] = "\(response.body.count)"
        headers["Connection"] = "close"

        var headerText = "HTTP/1.1 \(response.statusCode) \(response.reasonPhrase)\r\n"
        for (key, value) in headers {
            headerText += "\(key): \(value)\r\n"
        }
        headerText += "\r\n"

        var payload = Data(headerText.utf8)
        payload.append(response.body)

        connection.send(content: payload, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }

    static func normalizePath(_ path: String) -> String {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        if let queryIndex = trimmed.firstIndex(of: "?") {
            return String(trimmed[..<queryIndex])
        }
        return trimmed
    }

    static func normalizedPort(_ value: Int) -> Int {
        guard (1...65_535).contains(value) else { return 8080 }
        return value
    }

    static func reasonPhrase(for statusCode: Int) -> String {
        switch statusCode {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 403: return "Forbidden"
        case 404: return "Not Found"
        case 413: return "Payload Too Large"
        case 429: return "Too Many Requests"
        case 500: return "Internal Server Error"
        default: return "Error"
        }
    }

    static func json<T: Encodable>(statusCode: Int, payload: T) -> HTTPResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let body = (try? encoder.encode(payload)) ?? Data("{}".utf8)
        return HTTPResponse(
            statusCode: statusCode,
            reasonPhrase: reasonPhrase(for: statusCode),
            headers: ["Content-Type": "application/json; charset=utf-8"],
            body: body
        )
    }

    static func errorResponse(statusCode: Int, message: String, code: String?) -> HTTPResponse {
        json(
            statusCode: statusCode,
            payload: ErrorEnvelope(
                error: APIError(
                    message: message,
                    type: "invalid_request_error",
                    param: nil,
                    code: code
                )
            )
        )
    }

    static func finishReason(stopReason: String, completionTokens: Int, maxTokens: Int) -> String {
        let normalized = stopReason.lowercased()
        if normalized.contains("max") || completionTokens >= maxTokens {
            return "length"
        }
        return "stop"
    }
}

// MARK: - Payloads

private extension CLIOpenAICompatibleServer {
    struct ChatCompletionsRequest: Decodable {
        let model: String?
        let messages: [RequestMessage]
        let temperature: Double?
        let maxTokens: Int?
        let maxCompletionTokens: Int?
        let stream: Bool?
    }

    struct RequestMessage: Decodable {
        let role: String
        let content: RequestContent?

        var flattenedText: String {
            guard let content else { return "" }
            return content.flattenedText
        }
    }

    enum RequestContent: Decodable {
        case text(String)
        case parts([Part])

        struct Part: Decodable {
            let type: String?
            let text: String?
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let text = try? container.decode(String.self) {
                self = .text(text)
                return
            }
            if let parts = try? container.decode([Part].self) {
                self = .parts(parts)
                return
            }
            throw DecodingError.typeMismatch(
                RequestContent.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "content must be a string or an array of text parts"
                )
            )
        }

        var flattenedText: String {
            switch self {
            case .text(let value):
                return value
            case .parts(let parts):
                return parts.compactMap { part in
                    if let type = part.type, type != "text" {
                        return nil
                    }
                    return part.text
                }
                .joined(separator: "\n")
            }
        }
    }

    struct ModelsListResponse: Encodable {
        let object: String
        let data: [ModelDescriptor]
    }

    struct ModelDescriptor: Encodable {
        let id: String
        let object: String
        let created: Int
        let ownedBy: String

        enum CodingKeys: String, CodingKey {
            case id
            case object
            case created
            case ownedBy = "owned_by"
        }
    }

    struct ChatCompletionsResponse: Encodable {
        let id: String
        let object: String
        let created: Int
        let model: String
        let choices: [Choice]
        let usage: Usage
    }

    struct Choice: Encodable {
        let index: Int
        let message: ChoiceMessage
        let finishReason: String

        enum CodingKeys: String, CodingKey {
            case index
            case message
            case finishReason = "finish_reason"
        }
    }

    struct ChoiceMessage: Encodable {
        let role: String
        let content: String
    }

    struct Usage: Encodable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }

    struct ErrorEnvelope: Encodable {
        let error: APIError
    }

    struct APIError: Encodable {
        let message: String
        let type: String
        let param: String?
        let code: String?
    }
}

// MARK: - Network Helpers

private extension CLIOpenAICompatibleServer {
    static func remoteHostString(from endpoint: NWEndpoint) -> String? {
        guard case .hostPort(let host, _) = endpoint else { return nil }
        return host.debugDescription
    }

    static func isLoopbackHost(_ host: String) -> Bool {
        let normalized = host
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "[", with: "")
            .replacingOccurrences(of: "]", with: "")
            .lowercased()

        if normalized == "localhost" || normalized == "::1" {
            return true
        }
        if normalized.hasPrefix("127.") || normalized.hasPrefix("::ffff:127.") {
            return true
        }
        return false
    }

    static func preferredLANIPv4Address() -> String? {
        var pointer: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&pointer) == 0, let first = pointer else { return nil }
        defer { freeifaddrs(pointer) }

        var fallbackAddress: String?
        var current = first

        while true {
            let interface = current.pointee
            let flags = Int32(interface.ifa_flags)
            let isUp = (flags & IFF_UP) != 0
            let isRunning = (flags & IFF_RUNNING) != 0
            let isLoopback = (flags & IFF_LOOPBACK) != 0

            if isUp, isRunning, !isLoopback,
               let addr = interface.ifa_addr,
               addr.pointee.sa_family == UInt8(AF_INET) {
                let name = String(cString: interface.ifa_name)
                var hostname = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                if getnameinfo(
                    addr,
                    socklen_t(addr.pointee.sa_len),
                    &hostname,
                    socklen_t(hostname.count),
                    nil,
                    0,
                    NI_NUMERICHOST
                ) == 0 {
                    let ipBytes = hostname.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
                    let ip = String(decoding: ipBytes, as: UTF8.self)
                    if name.hasPrefix("en") {
                        return ip
                    }
                    if fallbackAddress == nil {
                        fallbackAddress = ip
                    }
                }
            }

            guard let next = interface.ifa_next else { break }
            current = next
        }

        return fallbackAddress
    }
}
