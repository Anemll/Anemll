//
//  OpenAICompatibleServerService.swift
//  ANEMLLChat
//
//  Lightweight OpenAI-compatible local HTTP server for the currently loaded model.
//

#if os(macOS) || os(tvOS)
import Foundation
import Network
import Darwin

enum OpenAICompatibleServerBindMode: String, CaseIterable, Identifiable, Sendable {
    case localhost
    case lan

    var id: String { rawValue }

    var title: String {
        switch self {
        case .localhost:
            return "Localhost Only"
        case .lan:
            return "LAN/WiFi"
        }
    }

    var summary: String {
        switch self {
        case .localhost:
            #if os(tvOS)
            return "Only apps on this Apple TV can connect."
            #else
            return "Only apps on this Mac can connect."
            #endif
        case .lan:
            return "Devices on your local network can connect."
        }
    }
}

@MainActor
final class OpenAICompatibleServerService: ObservableObject {
    static let shared = OpenAICompatibleServerService()

    @Published private(set) var isRunning = false
    @Published private(set) var lastErrorMessage: String?
    @Published private(set) var activeBindMode: OpenAICompatibleServerBindMode = .localhost
    @Published private(set) var activePort: Int = StorageService.defaultOpenAICompatibleServerPortValue
    @Published private(set) var localhostURL: String?
    @Published private(set) var lanURL: String?

    /// Reference to the model manager for listing/loading models via API
    weak var modelManager: ModelManagerViewModel?

    private let listenerQueue = DispatchQueue(label: "com.anemll.chat.openai-server")
    private var listener: NWListener?
    private var bindRetryTask: Task<Void, Never>?
    private var listenerToken: UInt64 = 0
    private var desiredConfig: ListenerConfig?
    private let maxAddressInUseRetries = 4

    private struct ListenerConfig: Equatable {
        let bindMode: OpenAICompatibleServerBindMode
        let port: Int
    }

    private init() {
        refreshURLs()
    }

    func restoreFromStorageAndApply() async {
        let enabled = await StorageService.shared.openAICompatibleServerEnabled
        let bindMode = await StorageService.shared.openAICompatibleServerBindMode
        let port = await StorageService.shared.openAICompatibleServerPort

        await applySettings(enabled: enabled, bindMode: bindMode, port: port)
    }

    func applySettings(enabled: Bool, bindMode: OpenAICompatibleServerBindMode, port: Int) async {
        let normalizedPort = Self.normalizedPort(port)
        await StorageService.shared.saveOpenAICompatibleServerEnabled(enabled)
        await StorageService.shared.saveOpenAICompatibleServerBindMode(bindMode)
        await StorageService.shared.saveOpenAICompatibleServerPort(normalizedPort)

        activeBindMode = bindMode
        activePort = normalizedPort
        refreshURLs()

        if enabled {
            start(bindMode: bindMode, port: normalizedPort)
        } else {
            stop()
        }
    }

    func stop() {
        desiredConfig = nil
        teardownListener(clearError: true, logStop: true, cancelRetryTask: true)
    }

    private func start(bindMode: OpenAICompatibleServerBindMode, port: Int) {
        let normalizedPort = Self.normalizedPort(port)
        let config = ListenerConfig(bindMode: bindMode, port: normalizedPort)
        desiredConfig = config
        activeBindMode = bindMode
        activePort = normalizedPort
        refreshURLs()
        lastErrorMessage = nil
        launchListener(with: config, attempt: 0)
    }

    private func launchListener(with config: ListenerConfig, attempt: Int) {
        guard desiredConfig == config else { return }
        teardownListener(clearError: false, logStop: false, cancelRetryTask: true)

        guard let nwPort = NWEndpoint.Port(rawValue: UInt16(config.port)) else {
            isRunning = false
            lastErrorMessage = "Invalid port \(config.port)."
            return
        }

        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true

        do {
            let listener: NWListener
            if config.bindMode == .localhost {
                params.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: nwPort)
                listener = try NWListener(using: params)
            } else {
                listener = try NWListener(using: params, on: nwPort)
            }

            listenerToken &+= 1
            let token = listenerToken

            listener.stateUpdateHandler = { [weak self] state in
                Task { @MainActor in
                    self?.handleListenerState(state, token: token, config: config, attempt: attempt)
                }
            }

            listener.newConnectionHandler = { [weak self] connection in
                guard let self else { return }
                connection.start(queue: self.listenerQueue)
                Self.receiveRequest(
                    on: connection,
                    accumulated: Data(),
                    mode: config.bindMode,
                    requestHandler: { request in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            await self.handleRequest(request, on: connection)
                        }
                    }
                )
            }

            self.listener = listener
            listener.start(queue: listenerQueue)

            if attempt == 0 {
                logInfo("OpenAI-compatible server starting on port \(config.port) (\(config.bindMode.rawValue))", category: .app)
            } else {
                logInfo("OpenAI-compatible server retrying start on port \(config.port) (\(config.bindMode.rawValue)) attempt \(attempt)/\(maxAddressInUseRetries)", category: .app)
            }
        } catch let nwError as NWError {
            if scheduleRetryIfNeeded(for: config, attempt: attempt, error: nwError) {
                return
            }
            isRunning = false
            lastErrorMessage = Self.errorMessage(for: nwError, port: config.port)
            logError("OpenAI-compatible server failed to start: \(nwError)", category: .app)
        } catch {
            isRunning = false
            lastErrorMessage = "Failed to start server: \(error.localizedDescription)"
            logError("OpenAI-compatible server failed to start: \(error)", category: .app)
        }
    }

    private func handleListenerState(
        _ state: NWListener.State,
        token: UInt64,
        config: ListenerConfig,
        attempt: Int
    ) {
        guard token == listenerToken else { return }

        switch state {
        case .setup:
            isRunning = false
        case .waiting(let error):
            isRunning = false
            if scheduleRetryIfNeeded(for: config, attempt: attempt, error: error) {
                return
            }
            lastErrorMessage = Self.errorMessage(for: error, port: config.port)
            logWarning("OpenAI-compatible server waiting: \(error)", category: .app)
        case .ready:
            isRunning = true
            lastErrorMessage = nil
            refreshURLs()
            logInfo("OpenAI-compatible server is ready on port \(activePort)", category: .app)
        case .failed(let error):
            isRunning = false
            listener = nil
            if scheduleRetryIfNeeded(for: config, attempt: attempt, error: error) {
                return
            }
            lastErrorMessage = Self.errorMessage(for: error, port: config.port)
            logError("OpenAI-compatible server failed: \(error)", category: .app)
        case .cancelled:
            isRunning = false
            listener = nil
        @unknown default:
            isRunning = false
        }
    }

    private func scheduleRetryIfNeeded(
        for config: ListenerConfig,
        attempt: Int,
        error: NWError
    ) -> Bool {
        guard Self.isAddressInUse(error) else { return false }
        guard attempt < maxAddressInUseRetries else { return false }
        guard desiredConfig == config else { return false }

        let nextAttempt = attempt + 1
        let delayMs = min(1_000, 200 * nextAttempt)

        logWarning(
            "OpenAI-compatible server port \(config.port) is busy; retrying in \(delayMs) ms (attempt \(nextAttempt)/\(maxAddressInUseRetries))",
            category: .app
        )

        bindRetryTask?.cancel()
        bindRetryTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(delayMs))
            guard let self else { return }
            guard !Task.isCancelled else { return }
            guard self.desiredConfig == config else { return }
            self.bindRetryTask = nil
            self.launchListener(with: config, attempt: nextAttempt)
        }
        return true
    }

    private func teardownListener(clearError: Bool, logStop: Bool, cancelRetryTask: Bool) {
        if cancelRetryTask {
            bindRetryTask?.cancel()
            bindRetryTask = nil
        }

        listenerToken &+= 1
        listener?.cancel()
        listener = nil
        isRunning = false

        if clearError {
            lastErrorMessage = nil
        }

        if logStop {
            logInfo("OpenAI-compatible server stopped", category: .app)
        }
    }

    private func refreshURLs() {
        localhostURL = "http://127.0.0.1:\(activePort)"

        guard activeBindMode == .lan else {
            lanURL = nil
            return
        }

        if let ip = Self.preferredLANIPv4Address() {
            lanURL = "http://\(ip):\(activePort)"
        } else {
            lanURL = nil
        }
    }

    private enum ChatCompletionsDispatch {
        case response(HTTPResponse)
        case streamed
    }

    private struct PreparedChatCompletionsRequest {
        let servedModelID: String
        let chatMessages: [ChatMessage]
    }

    private enum ParsedChatCompletionsRequest {
        case success(ChatCompletionsRequest)
        case failure(HTTPResponse)
    }

    private enum PreparedChatCompletionsResult {
        case success(PreparedChatCompletionsRequest)
        case failure(HTTPResponse)
    }

    private final class SSEStreamWriter: @unchecked Sendable {
        private let connection: NWConnection

        init(connection: NWConnection) {
            self.connection = connection
        }

        func sendJSON<T: Encodable>(_ payload: T) {
            OpenAICompatibleServerService.sendSSEJSONEvent(payload, on: connection)
        }

        func sendDone() {
            OpenAICompatibleServerService.sendSSEDone(on: connection)
        }
    }

    private func handleRequest(_ request: HTTPRequest, on connection: NWConnection) async {
        let path = Self.normalizePath(request.path)
        let debugLevel = await StorageService.shared.debugLevel
        let requestTraceID = String(UUID().uuidString.prefix(8))
        logIncomingRequest(request, normalizedPath: path, traceID: requestTraceID, debugLevel: debugLevel)

        switch (request.method, path) {
        case ("GET", "/v1/models"):
            let response = await handleModelsRequest()
            logOutgoingResponse(response, for: request, normalizedPath: path, traceID: requestTraceID, debugLevel: debugLevel)
            Self.send(response, on: connection)
        case ("POST", "/v1/chat/completions"):
            let dispatch = await processChatCompletionsRequest(
                body: request.body,
                on: connection,
                request: request,
                normalizedPath: path,
                traceID: requestTraceID,
                debugLevel: debugLevel
            )
            switch dispatch {
            case .response(let response):
                logOutgoingResponse(response, for: request, normalizedPath: path, traceID: requestTraceID, debugLevel: debugLevel)
                Self.send(response, on: connection)
            case .streamed:
                break
            }
        case ("GET", "/health"):
            let payload = ["status": "ok"]
            let response = Self.json(statusCode: 200, payload: payload)
            logOutgoingResponse(response, for: request, normalizedPath: path, traceID: requestTraceID, debugLevel: debugLevel)
            Self.send(response, on: connection)
        default:
            let response = Self.errorResponse(
                statusCode: 404,
                message: "Path not found: \(path)",
                code: "not_found"
            )
            logOutgoingResponse(response, for: request, normalizedPath: path, traceID: requestTraceID, debugLevel: debugLevel)
            Self.send(response, on: connection)
        }
    }

    private func processChatCompletionsRequest(
        body: Data,
        on connection: NWConnection,
        request: HTTPRequest,
        normalizedPath: String,
        traceID: String,
        debugLevel: Int
    ) async -> ChatCompletionsDispatch {
        let parsedRequest: ChatCompletionsRequest
        switch decodeChatCompletionsRequest(from: body) {
        case .success(let decoded):
            parsedRequest = decoded
        case .failure(let response):
            return .response(response)
        }

        if parsedRequest.stream == true {
            await handleStreamingChatCompletionsRequest(
                parsedRequest,
                on: connection,
                request: request,
                normalizedPath: normalizedPath,
                traceID: traceID,
                debugLevel: debugLevel
            )
            return .streamed
        }

        let response = await handleChatCompletionsRequest(parsedRequest)
        return .response(response)
    }

    private func decodeChatCompletionsRequest(from body: Data) -> ParsedChatCompletionsRequest {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        do {
            return .success(try decoder.decode(ChatCompletionsRequest.self, from: body))
        } catch {
            return .failure(
                Self.errorResponse(
                    statusCode: 400,
                    message: "Invalid JSON body: \(error.localizedDescription)",
                    code: "invalid_json"
                )
            )
        }
    }

    private func prepareChatCompletionsRequest(_ request: ChatCompletionsRequest) async -> PreparedChatCompletionsResult {
        let requestedModel = request.model?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        // If a specific model is requested and it differs from the loaded one, try to load it
        let currentModelID = await currentServedModelID()
        let needsModelSwitch = !requestedModel.isEmpty
            && requestedModel != "anemll-current-model"
            && (!InferenceService.shared.isModelLoaded || requestedModel != currentModelID)
        if needsModelSwitch {

            if let mgr = modelManager,
               let model = mgr.downloadedModels.first(where: { $0.id == requestedModel }) {

                if mgr.isLoadingModel {
                    return .failure(
                        Self.errorResponse(
                            statusCode: 429,
                            message: "Another model is currently loading. Please wait.",
                            code: "model_busy"
                        )
                    )
                }

                logInfo("[OpenAI Server] Auto-loading requested model: \(requestedModel)", category: .app)
                await mgr.loadModelForInference(model)

                guard InferenceService.shared.isModelLoaded else {
                    return .failure(
                        Self.errorResponse(
                            statusCode: 503,
                            message: "Failed to load model '\(requestedModel)'.",
                            code: "model_load_failed"
                        )
                    )
                }
            } else if InferenceService.shared.isModelLoaded {
                // Model is loaded but doesn't match — report mismatch
                let servedModelID = await currentServedModelID() ?? "anemll-current-model"
                return .failure(
                    Self.errorResponse(
                        statusCode: 400,
                        message: "Requested model '\(requestedModel)' is not downloaded. Active model: '\(servedModelID)'.",
                        code: "model_not_found"
                    )
                )
            }
        }

        guard InferenceService.shared.isModelLoaded else {
            return .failure(
                Self.errorResponse(
                    statusCode: 503,
                    message: "No model is currently loaded in ANEMLL Chat.",
                    code: "model_not_loaded"
                )
            )
        }

        if InferenceService.shared.isGenerating {
            return .failure(
                Self.errorResponse(
                    statusCode: 429,
                    message: "Model is busy generating another response.",
                    code: "model_busy"
                )
            )
        }

        let servedModelID = await currentServedModelID() ?? "anemll-current-model"

        var chatMessages: [ChatMessage] = []
        for message in request.messages {
            guard let role = message.mappedRole else {
                return .failure(
                    Self.errorResponse(
                        statusCode: 400,
                        message: "Unsupported message role '\(message.role)'.",
                        code: "invalid_role"
                    )
                )
            }

            let content = message.flattenedText.trimmingCharacters(in: .whitespacesAndNewlines)
            if content.isEmpty {
                continue
            }

            chatMessages.append(ChatMessage(role: role, content: content))
        }

        guard !chatMessages.isEmpty else {
            return .failure(
                Self.errorResponse(
                    statusCode: 400,
                    message: "Request must include at least one non-empty message.",
                    code: "empty_messages"
                )
            )
        }

        return .success(
            PreparedChatCompletionsRequest(
                servedModelID: servedModelID,
                chatMessages: chatMessages
            )
        )
    }

    private func withInferenceOverrides<T>(
        for request: ChatCompletionsRequest,
        _ operation: (_ inference: InferenceService, _ effectiveMaxTokens: Int) async throws -> T
    ) async throws -> T {
        let inference = InferenceService.shared
        let previousTemperature = inference.temperature
        let previousMaxTokens = inference.maxTokens
        let previousDoSample = inference.doSample
        var effectiveMaxTokens = inference.maxTokens

        if let requestTemperature = request.temperature {
            let clamped = Float(max(0.0, min(2.0, requestTemperature)))
            inference.temperature = clamped
            inference.doSample = clamped > 0
        }

        if let requestedMaxTokens = request.maxCompletionTokens ?? request.maxTokens {
            let clamped = max(1, min(requestedMaxTokens, inference.modelMaxContextSize))
            inference.maxTokens = clamped
            effectiveMaxTokens = clamped
        }

        defer {
            inference.temperature = previousTemperature
            inference.maxTokens = previousMaxTokens
            inference.doSample = previousDoSample
        }

        return try await operation(inference, effectiveMaxTokens)
    }

    private func handleStreamingChatCompletionsRequest(
        _ request: ChatCompletionsRequest,
        on connection: NWConnection,
        request rawRequest: HTTPRequest,
        normalizedPath: String,
        traceID: String,
        debugLevel: Int
    ) async {
        let prepared: PreparedChatCompletionsRequest
        switch await prepareChatCompletionsRequest(request) {
        case .success(let success):
            prepared = success
        case .failure(let errorResponse):
            logOutgoingResponse(errorResponse, for: rawRequest, normalizedPath: normalizedPath, traceID: traceID, debugLevel: debugLevel)
            Self.send(errorResponse, on: connection)
            return
        }

        guard await Self.sendSSEHeaders(on: connection) else {
            logError("[OpenAI Server] [\(traceID)] failed to start streaming response (header write failed)", category: .app)
            connection.cancel()
            return
        }

        if debugLevel >= 1 {
            logDebug(
                "[OpenAI Server] [\(traceID)] --> 200 \(rawRequest.method) \(normalizedPath) stream=true",
                category: .app
            )
        }

        let writer = SSEStreamWriter(connection: connection)
        let responseID = "chatcmpl-\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))"
        let created = Int(Date().timeIntervalSince1970)
        let streamModelID = prepared.servedModelID

        // OpenAI streaming responses typically begin with a role delta.
        writer.sendJSON(
            ChatCompletionsStreamResponse(
                id: responseID,
                object: "chat.completion.chunk",
                created: created,
                model: streamModelID,
                choices: [
                    StreamChoice(
                        index: 0,
                        delta: StreamChoiceDelta(role: "assistant", content: nil),
                        finishReason: nil
                    )
                ],
                usage: nil
            )
        )

        do {
            let (result, effectiveMaxTokens) = try await withInferenceOverrides(for: request) { inference, effectiveMaxTokens in
                let result = try await inference.generateResponse(
                    for: prepared.chatMessages,
                    onToken: { chunk in
                        guard !chunk.isEmpty else { return }
                        writer.sendJSON(
                            ChatCompletionsStreamResponse(
                                id: responseID,
                                object: "chat.completion.chunk",
                                created: created,
                                model: streamModelID,
                                choices: [
                                    StreamChoice(
                                        index: 0,
                                        delta: StreamChoiceDelta(role: nil, content: chunk),
                                        finishReason: nil
                                    )
                                ],
                                usage: nil
                            )
                        )
                    },
                    onWindowShift: { },
                    onHistoryUpdate: nil
                )
                return (result, effectiveMaxTokens)
            }

            let completionTokens = max(0, result.tokenCount)

            let finishReason = Self.finishReason(
                stopReason: result.stopReason,
                wasCancelled: result.wasCancelled,
                completionTokens: completionTokens,
                maxTokens: effectiveMaxTokens
            )

            writer.sendJSON(
                ChatCompletionsStreamResponse(
                    id: responseID,
                    object: "chat.completion.chunk",
                    created: created,
                    model: streamModelID,
                    choices: [
                        StreamChoice(
                            index: 0,
                            delta: StreamChoiceDelta(role: nil, content: nil),
                            finishReason: finishReason
                        )
                    ],
                    usage: nil
                )
            )
            writer.sendDone()

            if debugLevel >= 1 {
                logDebug(
                    "[OpenAI Server] [\(traceID)] stream completed tokens=\(completionTokens) finish=\(finishReason)",
                    category: .app
                )
            }
        } catch {
            logError("[OpenAI Server] [\(traceID)] streaming generation failed: \(error)", category: .app)
            writer.sendJSON(
                ErrorEnvelope(
                    error: APIError(
                        message: "Generation failed: \(error.localizedDescription)",
                        type: "invalid_request_error",
                        param: nil,
                        code: "generation_failed"
                    )
                )
            )
            writer.sendDone()
        }
    }

    private func logIncomingRequest(
        _ request: HTTPRequest,
        normalizedPath: String,
        traceID: String,
        debugLevel: Int
    ) {
        guard debugLevel >= 1 else { return }

        logDebug(
            "[OpenAI Server] [\(traceID)] <-- \(request.method) \(normalizedPath) bodyBytes=\(request.body.count)",
            category: .app
        )

        guard debugLevel >= 2 else { return }

        if request.path != normalizedPath {
            logDebug(
                "[OpenAI Server] [\(traceID)] normalizedPath '\(request.path)' -> '\(normalizedPath)'",
                category: .app
            )
        }

        if !request.headers.isEmpty {
            logDebug(
                "[OpenAI Server] [\(traceID)] requestHeaders \(Self.redactedHeadersDescription(request.headers))",
                category: .app
            )
        }

        if !request.body.isEmpty {
            logDebug(
                "[OpenAI Server] [\(traceID)] requestBody \(Self.bodyPreview(for: request.body, limit: 2_000))",
                category: .app
            )
        }
    }

    private func logOutgoingResponse(
        _ response: HTTPResponse,
        for request: HTTPRequest,
        normalizedPath: String,
        traceID: String,
        debugLevel: Int
    ) {
        guard debugLevel >= 1 else { return }

        logDebug(
            "[OpenAI Server] [\(traceID)] --> \(response.statusCode) \(request.method) \(normalizedPath) bodyBytes=\(response.body.count)",
            category: .app
        )

        guard debugLevel >= 2 else { return }

        if !response.headers.isEmpty {
            logDebug(
                "[OpenAI Server] [\(traceID)] responseHeaders \(Self.headersDescription(response.headers))",
                category: .app
            )
        }

        if !response.body.isEmpty {
            logDebug(
                "[OpenAI Server] [\(traceID)] responseBody \(Self.bodyPreview(for: response.body, limit: 2_000))",
                category: .app
            )
        }
    }

    private func handleModelsRequest() async -> HTTPResponse {
        let now = Int(Date().timeIntervalSince1970)

        // Return all downloaded models so clients can pick which to load
        if let mgr = modelManager {
            let downloaded = mgr.downloadedModels
            let data: [ModelDescriptor] = downloaded.map { model in
                ModelDescriptor(
                    id: model.id,
                    object: "model",
                    created: now,
                    ownedBy: "anemll"
                )
            }
            let payload = ModelsListResponse(object: "list", data: data)
            return Self.json(statusCode: 200, payload: payload)
        }

        // Fallback: no model manager wired up — report only the loaded model
        let modelId: String? = await currentServedModelID()
        let data: [ModelDescriptor]
        if let modelId {
            data = [ModelDescriptor(
                id: modelId,
                object: "model",
                created: now,
                ownedBy: "anemll"
            )]
        } else {
            data = []
        }

        let payload = ModelsListResponse(object: "list", data: data)
        return Self.json(statusCode: 200, payload: payload)
    }

    private func handleChatCompletionsRequest(_ request: ChatCompletionsRequest) async -> HTTPResponse {
        let prepared: PreparedChatCompletionsRequest
        switch await prepareChatCompletionsRequest(request) {
        case .success(let success):
            prepared = success
        case .failure(let errorResponse):
            return errorResponse
        }

        do {
            let (result, effectiveMaxTokens) = try await withInferenceOverrides(for: request) { inference, effectiveMaxTokens in
                let result = try await inference.generateResponse(
                    for: prepared.chatMessages,
                    onToken: { _ in },
                    onWindowShift: { },
                    onHistoryUpdate: nil
                )
                return (result, effectiveMaxTokens)
            }

            let promptTokens = max(0, result.prefillTokens)
            let completionTokens = max(0, result.tokenCount)
            let usage = Usage(
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                totalTokens: promptTokens + completionTokens
            )

            let finishReason = Self.finishReason(
                stopReason: result.stopReason,
                wasCancelled: result.wasCancelled,
                completionTokens: completionTokens,
                maxTokens: effectiveMaxTokens
            )

            let response = ChatCompletionsResponse(
                id: "chatcmpl-\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))",
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: prepared.servedModelID,
                choices: [
                    Choice(
                        index: 0,
                        message: ChoiceMessage(role: "assistant", content: result.text),
                        finishReason: finishReason
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

    private func currentServedModelID() async -> String? {
        guard InferenceService.shared.isModelLoaded else { return nil }

        if let selected = await StorageService.shared.selectedModelId,
           !selected.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return selected
        }

        if let modelId = InferenceService.shared.currentModelId,
           !modelId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return modelId
        }

        return "anemll-current-model"
    }

    private static func normalizedPort(_ port: Int) -> Int {
        guard (1...65_535).contains(port) else {
            return StorageService.defaultOpenAICompatibleServerPortValue
        }
        return port
    }

    private static func isAddressInUse(_ error: NWError) -> Bool {
        if case .posix(let code) = error {
            return code == .EADDRINUSE
        }
        return false
    }

    private static func errorMessage(for error: NWError, port: Int) -> String {
        if isAddressInUse(error) {
            return "Port \(port) is already in use. Stop the other server or choose a different port."
        }
        return "Server failed: \(error.localizedDescription)"
    }
}

// MARK: - HTTP Transport

private extension OpenAICompatibleServerService {
    nonisolated static let headerDelimiter = Data("\r\n\r\n".utf8)
    nonisolated static let maxRequestBytes = 4_194_304

    struct HTTPRequest {
        let method: String
        let path: String
        let headers: [String: String]
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

    nonisolated static func receiveRequest(
        on connection: NWConnection,
        accumulated: Data,
        mode: OpenAICompatibleServerBindMode,
        requestHandler: @escaping @Sendable (HTTPRequest) -> Void
    ) {
        if mode == .localhost,
           let remoteHost = remoteHostString(from: connection.endpoint),
           !isLoopbackHost(remoteHost) {
            let response = errorResponse(
                statusCode: 403,
                message: "Connection denied. Localhost mode accepts only loopback clients.",
                code: "forbidden_remote_host"
            )
            send(response, on: connection)
            return
        }

        connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { data, _, isComplete, error in
            var buffer = accumulated
            if let data, !data.isEmpty {
                buffer.append(data)
            }

            if buffer.count > maxRequestBytes {
                let response = errorResponse(
                    statusCode: 413,
                    message: "Request body is too large.",
                    code: "request_too_large"
                )
                send(response, on: connection)
                return
            }

            switch parseRequest(from: buffer) {
            case .complete(let request):
                requestHandler(request)

            case .malformed(let message):
                let response = errorResponse(
                    statusCode: 400,
                    message: message,
                    code: "bad_request"
                )
                send(response, on: connection)

            case .needMoreData:
                if let error {
                    let response = errorResponse(
                        statusCode: 400,
                        message: "Receive error: \(error.localizedDescription)",
                        code: "receive_error"
                    )
                    send(response, on: connection)
                    return
                }

                guard !isComplete else {
                    let response = errorResponse(
                        statusCode: 400,
                        message: "Connection closed before a complete request was received.",
                        code: "incomplete_request"
                    )
                    send(response, on: connection)
                    return
                }

                receiveRequest(
                    on: connection,
                    accumulated: buffer,
                    mode: mode,
                    requestHandler: requestHandler
                )
            }
        }
    }

    nonisolated static func parseRequest(from data: Data) -> ParseResult {
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

        var headers: [String: String] = [:]
        for line in lines.dropFirst() where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = line[..<colon].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines)
            headers[key] = value
        }

        let contentLength: Int
        if let rawLength = headers["content-length"], !rawLength.isEmpty {
            guard let parsed = Int(rawLength), parsed >= 0 else {
                return .malformed("Invalid Content-Length header.")
            }
            contentLength = parsed
        } else {
            contentLength = 0
        }

        let bodyStart = delimiterRange.upperBound
        let expectedTotal = bodyStart + contentLength
        guard data.count >= expectedTotal else {
            return .needMoreData
        }

        let body = Data(data[bodyStart..<expectedTotal])
        return .complete(HTTPRequest(method: method, path: path, headers: headers, body: body))
    }

    nonisolated static func send(_ response: HTTPResponse, on connection: NWConnection) {
        var headerText = "HTTP/1.1 \(response.statusCode) \(response.reasonPhrase)\r\n"
        var mergedHeaders = response.headers
        mergedHeaders["Content-Length"] = "\(response.body.count)"
        mergedHeaders["Connection"] = "close"

        for (key, value) in mergedHeaders {
            headerText += "\(key): \(value)\r\n"
        }
        headerText += "\r\n"

        var payload = Data(headerText.utf8)
        payload.append(response.body)

        connection.send(content: payload, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }

    nonisolated static func sendSSEHeaders(on connection: NWConnection) async -> Bool {
        let headerText = [
            "HTTP/1.1 200 OK",
            "Content-Type: text/event-stream; charset=utf-8",
            "Cache-Control: no-cache",
            "Connection: close",
            "X-Accel-Buffering: no",
            "",
            ""
        ].joined(separator: "\r\n")

        return await withCheckedContinuation { continuation in
            connection.send(content: Data(headerText.utf8), completion: .contentProcessed { error in
                continuation.resume(returning: error == nil)
            })
        }
    }

    nonisolated static func sendSSEJSONEvent<T: Encodable>(_ payload: T, on connection: NWConnection) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]

        guard let jsonData = try? encoder.encode(payload) else {
            return
        }

        var frame = Data("data: ".utf8)
        frame.append(jsonData)
        frame.append(Data("\n\n".utf8))
        sendSSEFrame(frame, on: connection, closeAfterSend: false)
    }

    nonisolated static func sendSSEDone(on connection: NWConnection) {
        sendSSEFrame(Data("data: [DONE]\n\n".utf8), on: connection, closeAfterSend: true)
    }

    nonisolated static func sendSSEFrame(_ frame: Data, on connection: NWConnection, closeAfterSend: Bool) {
        connection.send(content: frame, completion: .contentProcessed { error in
            if let error {
                logError("OpenAI-compatible SSE send failed: \(error)", category: .app)
            }
            if closeAfterSend {
                connection.cancel()
            }
        })
    }

    nonisolated static func normalizePath(_ rawPath: String) -> String {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let unqueriedPath: String

        if trimmed.hasPrefix("http://") || trimmed.hasPrefix("https://"),
           let url = URL(string: trimmed) {
            unqueriedPath = url.path.isEmpty ? "/" : url.path
        } else if let queryIndex = trimmed.firstIndex(of: "?") {
            unqueriedPath = String(trimmed[..<queryIndex])
        } else {
            unqueriedPath = trimmed
        }

        let components = unqueriedPath
            .split(separator: "/", omittingEmptySubsequences: true)
            .map(String.init)

        guard !components.isEmpty else {
            return "/"
        }

        return "/" + components.joined(separator: "/")
    }

    nonisolated static func reasonPhrase(for statusCode: Int) -> String {
        switch statusCode {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 403: return "Forbidden"
        case 404: return "Not Found"
        case 413: return "Payload Too Large"
        case 429: return "Too Many Requests"
        case 500: return "Internal Server Error"
        case 503: return "Service Unavailable"
        default: return "Error"
        }
    }

    nonisolated static func json<T: Encodable>(statusCode: Int, payload: T) -> HTTPResponse {
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

    nonisolated static func errorResponse(statusCode: Int, message: String, code: String?) -> HTTPResponse {
        let payload = ErrorEnvelope(
            error: APIError(
                message: message,
                type: "invalid_request_error",
                param: nil,
                code: code
            )
        )
        return json(statusCode: statusCode, payload: payload)
    }

    nonisolated static func bodyPreview(for data: Data, limit: Int) -> String {
        guard !data.isEmpty else { return "<empty>" }

        guard let utf8 = String(data: data, encoding: .utf8) else {
            return "<non-utf8 \(data.count) bytes>"
        }

        let flattened = utf8
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\n", with: "\\n")

        guard flattened.count > limit else { return flattened }
        let clipped = String(flattened.prefix(limit))
        return "\(clipped)… [truncated]"
    }

    nonisolated static func headersDescription(_ headers: [String: String]) -> String {
        headers
            .map { ($0.key, $0.value) }
            .sorted { lhs, rhs in lhs.0.lowercased() < rhs.0.lowercased() }
            .map { "\($0.0)=\($0.1)" }
            .joined(separator: " ")
    }

    nonisolated static func redactedHeadersDescription(_ headers: [String: String]) -> String {
        let sensitiveHeaders: Set<String> = [
            "authorization",
            "proxy-authorization",
            "cookie",
            "set-cookie",
            "x-api-key"
        ]

        let redacted = headers.map { key, value in
            let normalized = key.lowercased()
            if sensitiveHeaders.contains(normalized) {
                return (key, "<redacted>")
            }
            return (key, value)
        }

        return headersDescription(Dictionary(uniqueKeysWithValues: redacted))
    }

    nonisolated static func finishReason(
        stopReason: String,
        wasCancelled: Bool,
        completionTokens: Int,
        maxTokens: Int
    ) -> String {
        if wasCancelled {
            return "stop"
        }

        let normalized = stopReason.lowercased()
        if normalized.contains("max") || completionTokens >= maxTokens {
            return "length"
        }

        return "stop"
    }
}

// MARK: - OpenAI Payloads

private extension OpenAICompatibleServerService {
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

        var mappedRole: MessageRole? {
            switch role.lowercased() {
            case "system", "developer":
                return .system
            case "user":
                return .user
            case "assistant":
                return .assistant
            default:
                return nil
            }
        }

        var flattenedText: String {
            guard let content else { return "" }
            return content.flattenedText
        }
    }

    enum RequestContent: Decodable {
        case text(String)
        case parts([ContentPart])

        struct ContentPart: Decodable {
            let type: String?
            let text: String?
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let rawText = try? container.decode(String.self) {
                self = .text(rawText)
                return
            }
            if let parts = try? container.decode([ContentPart].self) {
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
            case .text(let raw):
                return raw
            case .parts(let parts):
                return parts
                    .compactMap { part in
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

    struct ChatCompletionsStreamResponse: Encodable {
        let id: String
        let object: String
        let created: Int
        let model: String
        let choices: [StreamChoice]
        let usage: Usage?
    }

    struct StreamChoice: Encodable {
        let index: Int
        let delta: StreamChoiceDelta
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index
            case delta
            case finishReason = "finish_reason"
        }
    }

    struct StreamChoiceDelta: Encodable {
        let role: String?
        let content: String?
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

private extension OpenAICompatibleServerService {
    nonisolated static func remoteHostString(from endpoint: NWEndpoint) -> String? {
        guard case .hostPort(let host, _) = endpoint else { return nil }
        return host.debugDescription
    }

    nonisolated static func isLoopbackHost(_ host: String) -> Bool {
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

    nonisolated static func preferredLANIPv4Address() -> String? {
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
                    let ip = String(cString: hostname)
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
#endif
