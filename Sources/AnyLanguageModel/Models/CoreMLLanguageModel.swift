#if CoreML
    import Foundation
    import CoreML
    import Tokenizers
    import JSONSchema
    @preconcurrency import Generation
    @preconcurrency import Models

    /// A language model that runs locally using Core ML.
    ///
    /// Use this model to run language models on-device with Core ML.
    /// The model must be compiled to `.mlmodelc` format before use.
    ///
    /// ```swift
    /// let modelURL = Bundle.main.url(
    ///     forResource: "MyModel",
    ///     withExtension: "mlmodelc"
    /// )!
    /// let model = try await CoreMLLanguageModel(url: modelURL)
    /// ```
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    public struct CoreMLLanguageModel: AnyLanguageModel.LanguageModel {
        /// The reason the model is unavailable.
        /// This model is always available.
        public typealias UnavailableReason = Never

        private let model: Models.LanguageModel
        private let tokenizer: any Tokenizer
        private let chatTemplateHandler: (@Sendable (Instructions?, Prompt) -> [Message])?
        private let toolsHandler: (@Sendable ([any Tool]) -> [ToolSpec])?

        /// Creates a Core ML language model.
        ///
        /// - Parameters:
        ///   - url: The URL to a compiled Core ML model (`.mlmodelc`).
        ///   - computeUnits: The compute units to use for inference.
        ///   - chatTemplateHandler: An optional handler to format chat messages.
        ///   - toolsHandler: An optional handler to convert tools to the model's expected format.
        ///
        /// - Throws: A `CoreMLLanguageModelError` if the model can't be loaded, the file doesn't exist, or the model is invalid.
        public init(
            url: URL,
            computeUnits: MLComputeUnits = .all,
            chatTemplateHandler: (@Sendable (Instructions?, Prompt) -> [Message])? = nil,
            toolsHandler: (@Sendable ([any Tool]) -> [ToolSpec])? = nil
        ) async throws {
            // Ensure the model is already compiled
            guard url.pathExtension == "mlmodelc" else {
                throw CoreMLLanguageModelError.compiledModelRequired
            }

            // Check if the file exists first
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw CoreMLLanguageModelError.modelNotFound(url)
            }

            do {
                // Load the model with the specified compute units
                self.model = try Models.LanguageModel.loadCompiled(url: url, computeUnits: computeUnits)
            } catch {
                // Map CoreML errors to our specific error cases
                throw CoreMLLanguageModelError.modelInvalid(url, underlyingError: error)
            }

            // Load the tokenizer
            self.tokenizer = try await model.tokenizer

            self.chatTemplateHandler = chatTemplateHandler
            self.toolsHandler = toolsHandler
        }

        public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            try validateNoImageSegments(in: session)

            if type != String.self {
                let jsonString = try await generateStructuredJSON(
                    session: session,
                    prompt: prompt,
                    schema: type.generationSchema,
                    options: options,
                    includeSchemaInPrompt: includeSchemaInPrompt
                )
                let generatedContent = try GeneratedContent(json: jsonString)
                let content = try type.init(generatedContent)
                return LanguageModelSession.Response(
                    content: content,
                    rawContent: generatedContent,
                    transcriptEntries: ArraySlice([])
                )
            }

            let toolSpecs = resolvedToolSpecs(for: session)
            let toolNames = Set(session.tools.map(\.name))
            var promptSource = makePromptSource(session: session, prompt: prompt)

            var visibleChunks: [String] = []
            var allEntries: [Transcript.Entry] = []
            var toolIteration = 0
            var previousToolCallSignature: String?

            // Generate, then keep looping for as long as the model answers with tool calls.
            while true {
                let tokens = try encodePrompt(promptSource, toolSpecs: toolSpecs)
                let generationConfig = toGenerationConfig(options, promptTokenCount: tokens.count)

                // Reset model state for new generation
                await model.resetState()

                let outputTokens = await model.generate(
                    config: generationConfig,
                    tokens: tokens,
                    model: model.callAsFunction
                )
                let assistantText = decodeAssistantText(from: outputTokens, promptTokenCount: tokens.count)

                // Tool calling requires the chat-template path: there is no other way to
                // render tool specs into the prompt or feed tool results back to the model.
                guard case .chat(var messages) = promptSource, !session.tools.isEmpty else {
                    visibleChunks.append(assistantText)
                    break
                }

                let parsed = CoreMLToolCallParser.parse(assistantText, knownToolNames: toolNames)
                visibleChunks.append(parsed.visibleText)

                if !parsed.calls.isEmpty {
                    toolIteration += 1
                    if toolIteration > Self.maximumToolIterations {
                        allEntries.append(
                            .toolCalls(Transcript.ToolCalls(makeTranscriptToolCalls(from: parsed.calls)))
                        )
                        throw Self.maxToolIterationsExceededError(limit: Self.maximumToolIterations)
                    }

                    // Guard against a model that keeps asking for the exact same tool call.
                    let signature = CoreMLToolCallParser.signature(for: parsed.calls)
                    if signature == previousToolCallSignature {
                        allEntries.append(
                            .toolCalls(Transcript.ToolCalls(makeTranscriptToolCalls(from: parsed.calls)))
                        )
                        throw Self.repeatedToolCallLoopError()
                    }
                    previousToolCallSignature = signature

                    let transcriptCalls = makeTranscriptToolCalls(from: parsed.calls)
                    let resolution = try await resolveToolCalls(transcriptCalls, session: session)
                    switch resolution {
                    case .stop(let calls):
                        if !calls.isEmpty {
                            allEntries.append(.toolCalls(Transcript.ToolCalls(calls)))
                        }
                        return LanguageModelSession.Response(
                            content: "" as! Content,
                            rawContent: GeneratedContent(""),
                            transcriptEntries: ArraySlice(allEntries)
                        )
                    case .invocations(let invocations):
                        if !invocations.isEmpty {
                            allEntries.append(.toolCalls(Transcript.ToolCalls(invocations.map(\.call))))

                            messages.append(
                                assistantToolCallMessage(
                                    text: parsed.visibleText,
                                    transcriptCalls: transcriptCalls,
                                    parsedCalls: parsed.calls
                                )
                            )
                            for invocation in invocations {
                                allEntries.append(.toolOutput(invocation.output))
                                messages.append(toolResultMessage(for: invocation.output))
                            }

                            promptSource = .chat(messages)
                            continue
                        }
                    }
                }

                break
            }

            let assistantText = visibleChunks.joined()
            return LanguageModelSession.Response(
                content: assistantText as! Content,
                rawContent: GeneratedContent(assistantText),
                transcriptEntries: ArraySlice(allEntries)
            )
        }

        public func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            // For now, only String is supported
            guard type == String.self else {
                return LanguageModelSession.ResponseStream(
                    stream: AsyncThrowingStream { continuation in
                        continuation.finish(
                            throwing: CoreMLLanguageModelError.structuredStreamingUnsupported
                        )
                    }
                )
            }

            // Validate that no image segments are present
            do {
                try validateNoImageSegments(in: session)
            } catch {
                return LanguageModelSession.ResponseStream(
                    stream: AsyncThrowingStream { continuation in
                        continuation.finish(throwing: error)
                    }
                )
            }

            // Transform the generation into ResponseStream snapshots
            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
                @Sendable continuation in
                let task = Task {
                    do {
                        let toolSpecs = resolvedToolSpecs(for: session)
                        let toolNames = Set(session.tools.map(\.name))
                        let toolsEnabled = !session.tools.isEmpty
                        var promptSource = makePromptSource(session: session, prompt: prompt)

                        var toolIteration = 0
                        var previousToolCallSignature: String?

                        // Generate, then keep looping for as long as the model answers with tool calls.
                        while true {
                            if Task.isCancelled { break }

                            let tokens = try encodePrompt(promptSource, toolSpecs: toolSpecs)
                            let generationConfig = toGenerationConfig(options, promptTokenCount: tokens.count)

                            await model.resetState()

                            let promptTokenCount = tokens.count
                            // Text already published to the transcript and to the stream for this turn.
                            var publishedText = ""

                            let outputTokens = await model.generate(
                                config: generationConfig,
                                tokens: tokens,
                                model: model.callAsFunction
                            ) { tokenIds in
                                if Task.isCancelled { return }

                                let assistantText = decodeAssistantText(
                                    from: tokenIds,
                                    promptTokenCount: promptTokenCount
                                )
                                // Hold back text that looks like the start of a tool call so that
                                // raw tool-call markup never reaches the transcript or the caller.
                                let visibleText =
                                    toolsEnabled
                                    ? CoreMLToolCallParser.visibleTextForStreaming(assistantText)
                                    : assistantText
                                guard visibleText != publishedText else { return }
                                publishedText = visibleText

                                // Grow the observable transcript so a Transcript-driven UI updates live.
                                session.growStreamingTranscript(text: visibleText)
                                continuation.yield(
                                    .init(
                                        content: (visibleText as! Content).asPartiallyGenerated(),
                                        rawContent: GeneratedContent(visibleText)
                                    )
                                )
                            }

                            if Task.isCancelled { break }

                            let assistantText = decodeAssistantText(
                                from: outputTokens,
                                promptTokenCount: promptTokenCount
                            )

                            // Tool calling requires the chat-template path: there is no other way to
                            // render tool specs into the prompt or feed tool results back to the model.
                            guard case .chat(var messages) = promptSource, toolsEnabled else {
                                publish(
                                    assistantText,
                                    ifDifferentFrom: publishedText,
                                    in: session,
                                    to: continuation
                                )
                                break
                            }

                            let parsed = CoreMLToolCallParser.parse(assistantText, knownToolNames: toolNames)

                            // Reconcile: the streaming heuristic is deliberately conservative, so the
                            // final parse is the authoritative split between prose and tool calls.
                            publish(
                                parsed.visibleText,
                                ifDifferentFrom: publishedText,
                                in: session,
                                to: continuation
                            )

                            if !parsed.calls.isEmpty {
                                toolIteration += 1
                                if toolIteration > Self.maximumToolIterations {
                                    session.appendTranscriptEntry(
                                        .toolCalls(
                                            Transcript.ToolCalls(makeTranscriptToolCalls(from: parsed.calls))
                                        )
                                    )
                                    throw Self.maxToolIterationsExceededError(
                                        limit: Self.maximumToolIterations
                                    )
                                }

                                // Guard against a model that keeps asking for the exact same tool call.
                                let signature = CoreMLToolCallParser.signature(for: parsed.calls)
                                if signature == previousToolCallSignature {
                                    session.appendTranscriptEntry(
                                        .toolCalls(
                                            Transcript.ToolCalls(makeTranscriptToolCalls(from: parsed.calls))
                                        )
                                    )
                                    throw Self.repeatedToolCallLoopError()
                                }
                                previousToolCallSignature = signature

                                let transcriptCalls = makeTranscriptToolCalls(from: parsed.calls)
                                let resolution = try await resolveToolCalls(transcriptCalls, session: session)
                                switch resolution {
                                case .stop(let calls):
                                    if !calls.isEmpty {
                                        session.appendTranscriptEntry(.toolCalls(Transcript.ToolCalls(calls)))
                                    }
                                    continuation.finish()
                                    return
                                case .invocations(let invocations):
                                    if !invocations.isEmpty {
                                        // Tool calls must land in the transcript before their outputs.
                                        session.appendTranscriptEntry(
                                            .toolCalls(Transcript.ToolCalls(invocations.map(\.call)))
                                        )

                                        messages.append(
                                            assistantToolCallMessage(
                                                text: parsed.visibleText,
                                                transcriptCalls: transcriptCalls,
                                                parsedCalls: parsed.calls
                                            )
                                        )
                                        for invocation in invocations {
                                            session.appendTranscriptEntry(.toolOutput(invocation.output))
                                            messages.append(toolResultMessage(for: invocation.output))
                                        }

                                        promptSource = .chat(messages)
                                        continue
                                    }
                                }
                            }

                            break
                        }

                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }

                continuation.onTermination = { _ in
                    task.cancel()
                }
            }

            return LanguageModelSession.ResponseStream(stream: stream)
        }

        /// Publishes `text` to the session transcript and the response stream when it differs
        /// from what was already published for the current turn.
        private func publish<Content>(
            _ text: String,
            ifDifferentFrom publishedText: String,
            in session: LanguageModelSession,
            to continuation: AsyncThrowingStream<
                LanguageModelSession.ResponseStream<Content>.Snapshot, any Error
            >.Continuation
        ) where Content: Generable {
            guard text != publishedText else { return }
            session.growStreamingTranscript(text: text)
            continuation.yield(
                .init(
                    content: (text as! Content).asPartiallyGenerated(),
                    rawContent: GeneratedContent(text)
                )
            )
        }

        // MARK: - Image Validation

        private func validateNoImageSegments(in session: LanguageModelSession) throws {
            // Note: Instructions is a plain text type without segments, so no image check needed there.
            // Check for image segments in the most recent prompt
            for entry in session.transcript.reversed() {
                if case .prompt(let p) = entry {
                    for segment in p.segments {
                        if case .image = segment {
                            throw CoreMLLanguageModelError.unsupportedFeature
                        }
                    }
                    break
                }
            }
        }
    }

    /// Errors that can occur when working with Core ML language models.
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    public enum CoreMLLanguageModelError: LocalizedError {
        /// The provided model isn't a compiled Core ML model.
        case compiledModelRequired

        /// The model file was not found at the specified URL.
        case modelNotFound(URL)

        /// The model file was found but is corrupted, incompatible, or otherwise invalid.
        case modelInvalid(URL, underlyingError: Error)
        /// Image segments are not supported in CoreMLLanguageModel
        case unsupportedFeature
        /// Structured response streaming is not supported in CoreMLLanguageModel
        case structuredStreamingUnsupported

        public var errorDescription: String? {
            switch self {
            case .compiledModelRequired:
                return
                    "A compiled Core ML model (.mlmodelc) is required. Please compile your model first using MLModel.compileModel(at:)."
            case .modelNotFound(let url):
                return "Core ML model not found at: \(url.path). Please verify the file exists and the path is correct."
            case .modelInvalid(let url, let underlyingError):
                return
                    "Core ML model at \(url.path) is invalid or corrupted: \(underlyingError.localizedDescription). Please verify the model file is valid and compatible with the current Core ML version."
            case .unsupportedFeature:
                return "This CoreMLLanguageModel does not support image segments"
            case .structuredStreamingUnsupported:
                return "This CoreMLLanguageModel does not support structured response streaming"
            }
        }
    }

    // MARK: -

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    extension CoreMLLanguageModel {
        /// Builds a generation config for a free-form (non-constrained) generation.
        ///
        /// - Important: `GenerationConfig.maxLength` defaults to 20 and `generate` stops as soon as
        ///   the total sequence reaches it, so it has to be sized against the prompt or nothing is
        ///   generated at all. `eosTokenId` likewise has no default, and without it generation never
        ///   stops early — it always runs to the token budget and decodes past the end of the turn,
        ///   which makes tool-call text impossible to parse reliably. The constrained-decoding path
        ///   in this file already sets both; this does the same for the plain path.
        private func toGenerationConfig(
            _ options: GenerationOptions,
            promptTokenCount: Int
        ) -> GenerationConfig {
            var config = toGenerationConfig(options)
            config.maxLength = config.maxNewTokens + promptTokenCount
            config.eosTokenId = tokenizer.eosTokenId
            config.bosTokenId = tokenizer.bosTokenId
            return config
        }

        private func toGenerationConfig(_ options: GenerationOptions) -> GenerationConfig {
            var config = GenerationConfig(maxNewTokens: options.maximumResponseTokens ?? 2048)

            // Map temperature
            if let temperature = options.temperature {
                config.temperature = Float(temperature)
            }

            // Map sampling mode
            if let sampling = options.sampling {
                switch sampling.mode {
                case .greedy:
                    config.doSample = false
                case .topK(let k, _):
                    config.doSample = true
                    config.topK = k
                case .nucleus(let p, _):
                    config.doSample = true
                    config.topP = Float(p)
                }
            }

            return config
        }

        private func toStructuredGenerationConfig(_ options: GenerationOptions) -> GenerationConfig {
            var config = GenerationConfig(maxNewTokens: options.maximumResponseTokens ?? 512)

            config.doSample = true
            if let temperature = options.temperature {
                config.temperature = Float(temperature)
            } else {
                config.temperature = 0.2
            }
            config.topP = 0.95
            config.repetitionPenalty = 1.1

            if let sampling = options.sampling {
                switch sampling.mode {
                case .greedy:
                    config.doSample = false
                case .topK(let k, _):
                    config.doSample = true
                    config.topK = k
                case .nucleus(let p, _):
                    config.doSample = true
                    config.topP = Float(p)
                }
            }

            return config
        }

        private func generateStructuredJSON(
            session: LanguageModelSession,
            prompt: Prompt,
            schema: GenerationSchema,
            options: GenerationOptions,
            includeSchemaInPrompt: Bool
        ) async throws -> String {
            let maxTokens = options.maximumResponseTokens ?? 512
            var generationConfig = toStructuredGenerationConfig(options)

            let promptTokens = try structuredPromptTokens(
                in: session,
                prompt: prompt,
                schema: schema,
                includeSchemaInPrompt: includeSchemaInPrompt
            )

            generationConfig.maxLength = generationConfig.maxNewTokens + promptTokens.count
            generationConfig.eosTokenId = tokenizer.eosTokenId
            generationConfig.bosTokenId = tokenizer.bosTokenId

            await model.resetState()

            let tokenTensor = MLTensor(promptTokens.map(Int32.init)).expandingShape(at: 0)
            let initialLogits = await model.predictNextTokenScores(tokenTensor, config: generationConfig)
            let endTokens: Set<Int> = []

            let backend = try CoreMLTokenBackend(
                model: model,
                tokenizer: tokenizer,
                config: generationConfig,
                tokens: promptTokens,
                initialLogits: initialLogits,
                maximumTokens: maxTokens,
                endTokens: endTokens
            )
            var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
            let json = try await generator.generate()
            return json
        }

        private func structuredPromptTokens(
            in session: LanguageModelSession,
            prompt: Prompt,
            schema: GenerationSchema,
            includeSchemaInPrompt: Bool
        ) throws -> [Int] {
            if let chatTemplateHandler = chatTemplateHandler {
                var messages = chatTemplateHandler(session.instructions, prompt)
                if includeSchemaInPrompt {
                    let schemaPrompt = schemaPrompt(for: schema)
                    if !schemaPrompt.isEmpty {
                        messages.insert(["role": "system", "content": schemaPrompt], at: 0)
                    }
                }
                let toolSpecs: [ToolSpec]? = toolsHandler?(session.tools)
                return try tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs)
            }

            var text = prompt.description
            if includeSchemaInPrompt {
                let schemaPrompt = schemaPrompt(for: schema)
                if !schemaPrompt.isEmpty {
                    text = "\(schemaPrompt)\n\n\(text)"
                }
            }
            return tokenizer.encode(text: text)
        }

        private func schemaPrompt(for schema: GenerationSchema) -> String {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            guard
                let data = try? encoder.encode(schema),
                let jsonSchema = try? JSONDecoder().decode(JSONSchema.self, from: data),
                let schemaJSON = String(data: data, encoding: .utf8)
            else {
                return schema.schemaPrompt()
            }

            var header = "Respond with valid JSON matching this \(jsonSchema.typeName) schema"
            if let description = jsonSchema.description, !description.isEmpty {
                header += " (\(description))"
            }

            if let constValue = jsonSchema.const,
                let data = try? encoder.encode(constValue),
                let constString = String(data: data, encoding: .utf8)
            {
                header += ". Expected value: \(constString)"
            } else if let enumValues = jsonSchema.enum, !enumValues.isEmpty,
                let data = try? encoder.encode(JSONValue.array(enumValues)),
                let enumString = String(data: data, encoding: .utf8)
            {
                header += ". Allowed values: \(enumString)"
            }

            return "\(header):\n\(schemaJSON)"
        }

        private struct CoreMLTokenBackend: TokenBackend {
            let model: Models.LanguageModel
            let tokenizer: any Tokenizer
            let config: GenerationConfig
            let logitsProcessorList: LogitsProcessorList
            let endTokens: Set<Int>
            let eosToken: Int
            let vocabSize: Int

            var tokens: [Int]
            var currentLogits: MLTensor
            var remainingTokens: Int
            let totalTokenBudget: Int

            init(
                model: Models.LanguageModel,
                tokenizer: any Tokenizer,
                config: GenerationConfig,
                tokens: [Int],
                initialLogits: MLTensor,
                maximumTokens: Int,
                endTokens: Set<Int>
            ) throws {
                self.model = model
                self.tokenizer = tokenizer
                self.config = config
                self.tokens = tokens
                self.currentLogits = initialLogits
                self.remainingTokens = maximumTokens
                self.totalTokenBudget = maximumTokens
                self.endTokens = endTokens
                self.eosToken = config.eosTokenId ?? tokenizer.eosTokenId ?? 0
                self.vocabSize = initialLogits.shape.last ?? 0
                self.logitsProcessorList = CoreMLLanguageModel.makeLogitsProcessorList(config: config)
            }

            func tokenize(_ text: String) throws -> [Int] {
                tokenizer.encode(text: text, addSpecialTokens: false)
            }

            func tokenText(_ token: Int) -> String? {
                let decoded = tokenizer.decode(tokens: [token], skipSpecialTokens: false)
                return decoded.isEmpty ? nil : decoded
            }

            func isSpecialToken(_ token: Int) -> Bool {
                let raw = tokenizer.decode(tokens: [token], skipSpecialTokens: false)
                guard !raw.isEmpty else { return false }
                let filtered = tokenizer.decode(tokens: [token], skipSpecialTokens: true)
                return filtered.isEmpty
            }

            mutating func decode(_ token: Int) async throws {
                tokens.append(token)
                remainingTokens -= 1
                let tokenTensor = MLTensor(tokens.map(Int32.init)).expandingShape(at: 0)
                currentLogits = await model.predictNextTokenScores(tokenTensor, config: config)
            }

            mutating func sample(from allowedTokens: Set<Int>) async throws -> Int {
                guard !allowedTokens.isEmpty else {
                    throw ConstrainedGenerationError.tokenizationFailed
                }

                // Run logits processors on Float32 scores for stable behavior
                let inputIds = MLTensor(tokens.map(Int32.init)).expandingShape(at: 0)
                let floatScores =
                    currentLogits.scalarType == Float.self
                    ? currentLogits
                    : currentLogits.cast(to: Float.self)
                let vocabSize = floatScores.shape.last ?? self.vocabSize

                // Build a mask tensor that keeps only the allowed tokens.
                var maskValues = Array(repeating: -Float.infinity, count: vocabSize)
                var hasValidToken = false
                for token in allowedTokens {
                    if token >= 0 && token < vocabSize {
                        maskValues[token] = 0
                        hasValidToken = true
                    }
                }
                guard hasValidToken else {
                    throw ConstrainedGenerationError.tokenizationFailed
                }
                let maskTensor = MLTensor(maskValues).reshaped(to: floatScores.shape)
                let maskedScores = floatScores + maskTensor
                let processedScores = await logitsProcessorList(inputIds, maskedScores)

                let tokenTensor: MLTensor
                if config.doSample {
                    // Multinomial sample from candidate probabilities
                    let probs = processedScores.softmax(alongAxis: -1)
                    let prefixShape = Array(processedScores.shape.dropLast())
                    let randomShape = prefixShape + [1]
                    let rndTensor = MLTensor(randomUniform: randomShape, in: 0 ..< 1, scalarType: Float.self)
                    let cumulativeProbs = probs.cumulativeSum(alongAxis: -1)
                    let rnd =
                        cumulativeProbs.scalarType == Float.self
                        ? rndTensor : rndTensor.cast(to: cumulativeProbs.scalarType)

                    let mask = cumulativeProbs .< rnd
                    let penalized = mask * 1000.0
                    let indexed = penalized + cumulativeProbs
                    let sampledIndex = indexed.argmin(alongAxis: -1)
                    tokenTensor =
                        sampledIndex.scalarType == Int32.self ? sampledIndex : sampledIndex.cast(to: Int32.self)
                } else {
                    // Greedy select the best-scoring candidate
                    let selectedIndex = processedScores.argmax(alongAxis: -1)
                    tokenTensor =
                        selectedIndex.scalarType == Int32.self ? selectedIndex : selectedIndex.cast(to: Int32.self)
                }

                // Materialize the chosen token id
                let tokenArray = await tokenTensor.shapedArray(of: Int32.self)
                guard let token = tokenArray.scalars.last else {
                    throw ConstrainedGenerationError.tokenizationFailed
                }

                return Int(token)
            }
        }

        fileprivate static func makeLogitsProcessorList(config: GenerationConfig) -> LogitsProcessorList {
            var processors: [any LogitsProcessor] = []

            if config.repetitionPenalty != 1.0 {
                if let processor = try? RepetitionPenaltyLogitsProcessor(penalty: Float(config.repetitionPenalty)) {
                    processors.append(processor)
                }
            }

            if config.temperature > 0 && config.temperature != 1.0 {
                if let processor = try? TemperatureLogitsWarper(temperature: config.temperature) {
                    processors.append(processor)
                }
            }

            if config.topK > 0 && config.topK < Int.max {
                if let processor = try? TopKLogitsWarper(topK: config.topK) {
                    processors.append(processor)
                }
            }

            if config.topP < 1.0 {
                if let processor = try? TopPLogitsWarper(topP: Float(config.topP)) {
                    processors.append(processor)
                }
            }

            if let minP = config.minP {
                if let processor = try? MinPLogitsWarper(minP: Float(minP)) {
                    processors.append(processor)
                }
            }

            return LogitsProcessorList(processors: processors)
        }

    }

    // MARK: - Tool Calling

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    extension CoreMLLanguageModel {
        /// The maximum number of tool round-trips allowed for a single response.
        fileprivate static var maximumToolIterations: Int { 8 }

        fileprivate static func maxToolIterationsExceededError(
            limit: Int
        ) -> LanguageModelSession.GenerationError {
            .decodingFailure(
                .init(
                    debugDescription:
                        "Exceeded maximum tool iterations (\(limit)) while processing Core ML tool calls."
                )
            )
        }

        fileprivate static func repeatedToolCallLoopError() -> LanguageModelSession.GenerationError {
            .decodingFailure(
                .init(
                    debugDescription:
                        "Detected repeated Core ML tool-call signature and aborted to avoid an infinite tool loop."
                )
            )
        }

        // MARK: Prompt construction

        /// How the prompt for a generation turn is expressed.
        fileprivate enum PromptSource {
            /// Chat messages rendered through the tokenizer's chat template. Tool calling is only
            /// available on this path.
            case chat([Message])
            /// Raw text encoded directly by the tokenizer, with no chat template involved.
            case rawText(String)
        }

        /// Resolves the tool specifications handed to the chat template.
        ///
        /// A caller-supplied `toolsHandler` always wins, and is still called exactly as before —
        /// including when the session has no tools. When no handler is supplied, the session's tools
        /// are converted with the conventional OpenAI-style function schema that Hugging Face chat
        /// templates expect, instead of silently dropping them.
        fileprivate func resolvedToolSpecs(for session: LanguageModelSession) -> [ToolSpec]? {
            if let toolsHandler {
                return toolsHandler(session.tools)
            }
            guard !session.tools.isEmpty else { return nil }
            return session.tools.map { convertToolToToolSpec($0) }
        }

        fileprivate func makePromptSource(
            session: LanguageModelSession,
            prompt: Prompt
        ) -> PromptSource {
            if let chatTemplateHandler {
                return .chat(chatTemplateHandler(session.instructions, prompt))
            }

            // Without a chat template handler the model is normally prompted with raw text, but tool
            // specs and tool results can only be expressed through the chat template. When the
            // session actually has tools, build the minimal message list the handler would have
            // produced rather than dropping the tools.
            if !session.tools.isEmpty, tokenizer.hasChatTemplate {
                var messages: [Message] = []
                if let instructions = session.instructions {
                    let content = instructions.description
                    if !content.isEmpty {
                        messages.append(["role": "system", "content": content])
                    }
                }
                messages.append(["role": "user", "content": prompt.description])
                return .chat(messages)
            }

            return .rawText(prompt.description)
        }

        fileprivate func encodePrompt(
            _ source: PromptSource,
            toolSpecs: [ToolSpec]?
        ) throws -> [Int] {
            switch source {
            case .chat(let messages):
                return try tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs)
            case .rawText(let text):
                return tokenizer.encode(text: text)
            }
        }

        /// Strips the prompt at the token level to avoid issues with normalization or whitespace
        /// differences in decoded strings.
        fileprivate func decodeAssistantText(from outputTokens: [Int], promptTokenCount: Int) -> String {
            let assistantTokenSlice: ArraySlice<Int>
            if outputTokens.count >= promptTokenCount {
                assistantTokenSlice = outputTokens.dropFirst(promptTokenCount)
            } else {
                // Fallback: if the model did not echo the full prompt,
                // treat the entire output as assistant tokens
                assistantTokenSlice = outputTokens[outputTokens.indices]
            }
            return tokenizer.decode(tokens: Array(assistantTokenSlice))
        }

        // MARK: Tool specs

        /// Converts a tool to the OpenAI-style function schema used by Hugging Face chat templates.
        private func convertToolToToolSpec(_ tool: any Tool) -> ToolSpec {
            let parametersDict: [String: any Sendable]
            do {
                let resolvedSchema = tool.parameters.withResolvedRoot() ?? tool.parameters
                let data = try JSONEncoder().encode(resolvedSchema)
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    parametersDict = CoreMLToolCallParser.sendableJSONObject(from: json)
                } else {
                    parametersDict = CoreMLToolCallParser.emptyJSONSchemaObject
                }
            } catch {
                parametersDict = CoreMLToolCallParser.emptyJSONSchemaObject
            }

            let functionSpec: [String: any Sendable] = [
                "name": tool.name,
                "description": tool.description,
                "parameters": parametersDict,
            ]

            return [
                "type": "function",
                "function": functionSpec,
            ]
        }

        // MARK: Feeding results back to the model

        /// The assistant turn that requested the tools.
        ///
        /// The tool calls are expressed as structured `tool_calls`, which is what the Hermes/Qwen,
        /// Llama 3.x and Mistral chat templates read. `content` holds only the model's prose so that
        /// templates rendering both `content` and `tool_calls` do not emit the call twice.
        fileprivate func assistantToolCallMessage(
            text: String,
            transcriptCalls: [Transcript.ToolCall],
            parsedCalls: [CoreMLToolCallParser.ParsedToolCall]
        ) -> Message {
            var toolCalls: [[String: any Sendable]] = []
            toolCalls.reserveCapacity(parsedCalls.count)
            for (transcriptCall, parsedCall) in zip(transcriptCalls, parsedCalls) {
                let function: [String: any Sendable] = [
                    "name": parsedCall.name,
                    "arguments": parsedCall.arguments,
                ]
                toolCalls.append([
                    "id": transcriptCall.id,
                    "type": "function",
                    "function": function,
                ])
            }

            return [
                "role": "assistant",
                "content": text,
                "tool_calls": toolCalls,
            ]
        }

        /// The tool result turn.
        ///
        /// `role: "tool"` is what the Hermes/Qwen and Mistral templates expect; the Llama 3.x
        /// templates accept either `"tool"` or `"ipython"`. `tool_call_id` and `name` are extra keys
        /// that templates which do not use them simply ignore.
        fileprivate func toolResultMessage(for output: Transcript.ToolOutput) -> Message {
            [
                "role": "tool",
                "tool_call_id": output.id,
                "name": output.toolName,
                "content": toolOutputText(output),
            ]
        }

        private func toolOutputText(_ output: Transcript.ToolOutput) -> String {
            var textParts: [String] = []
            for segment in output.segments {
                switch segment {
                case .text(let textSegment):
                    textParts.append(textSegment.content)
                case .structure(let structuredSegment):
                    textParts.append(structuredSegment.content.jsonString)
                case .image:
                    // Image segments are not supported in Core ML tool output.
                    break
                }
            }
            return textParts.joined(separator: "\n")
        }

        // MARK: Tool invocation

        fileprivate struct ToolInvocationResult {
            let call: Transcript.ToolCall
            let output: Transcript.ToolOutput
        }

        fileprivate enum ToolResolutionOutcome {
            case stop(calls: [Transcript.ToolCall])
            case invocations([ToolInvocationResult])
        }

        fileprivate func makeTranscriptToolCalls(
            from parsedCalls: [CoreMLToolCallParser.ParsedToolCall]
        ) -> [Transcript.ToolCall] {
            parsedCalls.map { parsedCall in
                let arguments =
                    (try? GeneratedContent(json: parsedCall.argumentsJSON))
                    ?? GeneratedContent(kind: .structure(properties: [:], orderedKeys: []))
                return Transcript.ToolCall(
                    id: CoreMLToolCallParser.makeToolCallID(),
                    toolName: parsedCall.name,
                    arguments: arguments
                )
            }
        }

        // NOTE: Every provider keeps its own file-private `resolveToolCalls`. This is Core ML's copy
        // of that shared shape, written here because the existing ones are private to their files.
        fileprivate func resolveToolCalls(
            _ transcriptCalls: [Transcript.ToolCall],
            session: LanguageModelSession
        ) async throws -> ToolResolutionOutcome {
            guard !transcriptCalls.isEmpty else { return .invocations([]) }

            var toolsByName: [String: any Tool] = [:]
            for tool in session.tools where toolsByName[tool.name] == nil {
                toolsByName[tool.name] = tool
            }

            if let delegate = session.toolExecutionDelegate {
                await delegate.didGenerateToolCalls(transcriptCalls, in: session)
            }

            var decisions: [ToolExecutionDecision] = []
            decisions.reserveCapacity(transcriptCalls.count)

            if let delegate = session.toolExecutionDelegate {
                for call in transcriptCalls {
                    let decision = await delegate.toolCallDecision(for: call, in: session)
                    if case .stop = decision {
                        return .stop(calls: transcriptCalls)
                    }
                    decisions.append(decision)
                }
            } else {
                decisions = Array(repeating: .execute, count: transcriptCalls.count)
            }

            var results: [ToolInvocationResult] = []
            results.reserveCapacity(transcriptCalls.count)

            for (index, call) in transcriptCalls.enumerated() {
                switch decisions[index] {
                case .stop:
                    // This branch should be unreachable because `.stop` returns during decision
                    // collection. Keep it as a defensive guard in case that logic changes.
                    return .stop(calls: transcriptCalls)
                case .provideOutput(let segments):
                    let output = Transcript.ToolOutput(
                        id: call.id,
                        toolName: call.toolName,
                        segments: segments
                    )
                    if let delegate = session.toolExecutionDelegate {
                        await delegate.didExecuteToolCall(call, output: output, in: session)
                    }
                    results.append(ToolInvocationResult(call: call, output: output))
                case .execute:
                    guard let tool = toolsByName[call.toolName] else {
                        let message = Transcript.Segment.text(
                            .init(content: "Tool not found: \(call.toolName)")
                        )
                        let output = Transcript.ToolOutput(
                            id: call.id,
                            toolName: call.toolName,
                            segments: [message]
                        )
                        if let delegate = session.toolExecutionDelegate {
                            await delegate.didExecuteToolCall(call, output: output, in: session)
                        }
                        results.append(ToolInvocationResult(call: call, output: output))
                        continue
                    }

                    do {
                        let segments = try await tool.makeOutputSegments(from: call.arguments)
                        let output = Transcript.ToolOutput(
                            id: call.id,
                            toolName: tool.name,
                            segments: segments
                        )
                        if let delegate = session.toolExecutionDelegate {
                            await delegate.didExecuteToolCall(call, output: output, in: session)
                        }
                        results.append(ToolInvocationResult(call: call, output: output))
                    } catch {
                        if let delegate = session.toolExecutionDelegate {
                            await delegate.didFailToolCall(call, error: error, in: session)
                        }
                        throw LanguageModelSession.ToolCallError(tool: tool, underlyingError: error)
                    }
                }
            }

            return .invocations(results)
        }
    }

    // MARK: - Tool Call Parsing

    /// Parses tool calls out of the plain text a Core ML hosted model generates.
    ///
    /// Core ML models have no structured tool-call channel. `swift-transformers` renders tool specs
    /// into the prompt through the tokenizer's Jinja chat template and stops there — it has no
    /// tool-call parser at all — so the model's request comes back as ordinary generated text in
    /// whatever format its chat template taught it. Parsing is therefore necessarily
    /// format-specific, and this parser supports an explicit, closed set of formats:
    ///
    /// 1. Hermes / Qwen / NousResearch tags: `<tool_call>{"name": …, "arguments": {…}}</tool_call>`,
    ///    repeated for multiple calls. `<function_call>…</function_call>` is accepted as a synonym.
    /// 2. Mistral: `[TOOL_CALLS] [{"name": …, "arguments": {…}}]`.
    /// 3. Llama 3.x: `<|python_tag|>{"name": …, "parameters": {…}}`, with `;` separating calls.
    /// 4. A fenced ```` ```json ```` block whose content has tool-call shape.
    /// 5. A bare JSON object or array that has tool-call shape and nothing else in the turn.
    ///
    /// Formats 4 and 5 are ambiguous with a model simply answering in JSON, so they only count as a
    /// tool call when the name matches a tool the session actually has. Formats 1 to 3 are keyed on
    /// unambiguous markers, so an unknown name there is reported as a call to a missing tool.
    ///
    /// Not supported: DeepSeek's `<｜tool▁calls▁begin｜>` markers, Llama's `<|python_tag|>` when it
    /// carries actual Python rather than JSON, and any XML-argument format such as Claude's
    /// `<invoke>` blocks.
    /// - Note: Internal rather than file-private only so that the parsing rules can be unit tested
    ///   without a downloaded Core ML model.
    enum CoreMLToolCallParser {
        struct ParsedToolCall {
            let name: String
            /// Canonical (sorted-key) JSON text of the arguments object.
            let argumentsJSON: String
            /// The same arguments, ready to be handed back to the chat template.
            let arguments: [String: any Sendable]
        }

        struct ParseResult {
            /// The generated text with tool-call markup removed.
            let visibleText: String
            let calls: [ParsedToolCall]
        }

        static let emptyJSONSchemaObject: [String: any Sendable] = [
            "type": "object",
            "properties": [String: any Sendable](),
            "required": [String](),
        ]

        /// Paired tags that unambiguously wrap a tool call.
        private static let tagPairs = [
            ("<tool_call>", "</tool_call>"),
            ("<function_call>", "</function_call>"),
        ]

        /// Markers that introduce a tool call and run to the end of the turn.
        private static let leadingMarkers = ["[TOOL_CALLS]", "<|python_tag|>"]

        /// Everything that signals a tool call may be starting, used to hold text back mid-stream.
        private static var streamingMarkers: [String] {
            tagPairs.map(\.0) + leadingMarkers
        }

        // MARK: Streaming

        /// The portion of a partially generated turn that is safe to show the caller.
        ///
        /// This runs on every token, so it stays cheap and conservative: text is cut at the first
        /// tool-call marker, and a turn that starts with `{` or `[` is withheld entirely because it
        /// may still turn out to be a bare-JSON tool call. A fenced ```` ```json ```` tool call is
        /// not withheld and is therefore briefly visible before ``parse(_:knownToolNames:)`` removes
        /// it at the end of the turn.
        static func visibleTextForStreaming(_ text: String) -> String {
            var cutoff = text.endIndex
            for marker in streamingMarkers {
                if let range = text.range(of: marker), range.lowerBound < cutoff {
                    cutoff = range.lowerBound
                }
            }

            let visibleText = String(text[text.startIndex ..< cutoff])
            let trimmed = visibleText.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("{") || trimmed.hasPrefix("[") {
                return ""
            }
            return visibleText
        }

        // MARK: Parsing

        static func parse(_ text: String, knownToolNames: Set<String>) -> ParseResult {
            if let result = parseTaggedBlocks(text) { return result }
            if let result = parseLeadingMarker(text) { return result }
            if let result = parseFencedJSON(text, knownToolNames: knownToolNames) { return result }
            if let result = parseBareJSON(text, knownToolNames: knownToolNames) { return result }
            return ParseResult(visibleText: text, calls: [])
        }

        /// A stable identity for a set of calls, used to detect a model looping on the same request.
        static func signature(for calls: [ParsedToolCall]) -> String {
            calls.map { "\($0.name):\($0.argumentsJSON)" }.joined(separator: "|")
        }

        /// Mistral's chat template rejects tool call ids that are not exactly nine alphanumeric
        /// characters, and no other template cares about the shape, so use that everywhere.
        static func makeToolCallID() -> String {
            let alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            return String((0 ..< 9).compactMap { _ in alphabet.randomElement() })
        }

        // MARK: Format-specific parsing

        private static func parseTaggedBlocks(_ text: String) -> ParseResult? {
            var calls: [ParsedToolCall] = []
            var visibleText = ""
            var index = text.startIndex

            while index < text.endIndex {
                var earliest: (open: Range<String.Index>, close: String)?
                for (open, close) in tagPairs {
                    guard let range = text.range(of: open, range: index ..< text.endIndex) else { continue }
                    if earliest == nil || range.lowerBound < earliest!.open.lowerBound {
                        earliest = (range, close)
                    }
                }
                guard let (openRange, closeTag) = earliest else { break }

                visibleText += text[index ..< openRange.lowerBound]

                let bodyStart = openRange.upperBound
                let bodyEnd: String.Index
                if let closeRange = text.range(of: closeTag, range: bodyStart ..< text.endIndex) {
                    bodyEnd = closeRange.lowerBound
                    index = closeRange.upperBound
                } else {
                    // Truncated generation: take the rest of the turn as the call body.
                    bodyEnd = text.endIndex
                    index = text.endIndex
                }

                calls.append(contentsOf: toolCalls(fromJSONText: String(text[bodyStart ..< bodyEnd])))
            }

            // A tag with an unparseable body is treated as ordinary text rather than a lost call.
            guard !calls.isEmpty else { return nil }

            visibleText += text[index ..< text.endIndex]
            return ParseResult(
                visibleText: visibleText.trimmingCharacters(in: .whitespacesAndNewlines),
                calls: calls
            )
        }

        private static func parseLeadingMarker(_ text: String) -> ParseResult? {
            var earliest: Range<String.Index>?
            for marker in leadingMarkers {
                guard let range = text.range(of: marker) else { continue }
                if earliest == nil || range.lowerBound < earliest!.lowerBound {
                    earliest = range
                }
            }
            guard let markerRange = earliest else { return nil }

            let body = String(text[markerRange.upperBound...])
            var calls = toolCalls(fromJSONText: body)
            if calls.isEmpty {
                // Llama 3.1 separates multiple calls with `;`.
                calls = body.split(separator: ";").flatMap { toolCalls(fromJSONText: String($0)) }
            }
            guard !calls.isEmpty else { return nil }

            let visibleText = String(text[text.startIndex ..< markerRange.lowerBound])
            return ParseResult(
                visibleText: visibleText.trimmingCharacters(in: .whitespacesAndNewlines),
                calls: calls
            )
        }

        private static func parseFencedJSON(
            _ text: String,
            knownToolNames: Set<String>
        ) -> ParseResult? {
            let fence = "```"
            var calls: [ParsedToolCall] = []
            var visibleText = ""
            var index = text.startIndex

            while index < text.endIndex {
                guard let openRange = text.range(of: fence, range: index ..< text.endIndex),
                    let closeRange = text.range(of: fence, range: openRange.upperBound ..< text.endIndex)
                else { break }

                var body = String(text[openRange.upperBound ..< closeRange.lowerBound])
                // Drop an optional language tag on the opening fence.
                if let newline = body.firstIndex(of: "\n") {
                    let firstLine = body[body.startIndex ..< newline].trimmingCharacters(
                        in: .whitespaces
                    )
                    if firstLine.isEmpty || firstLine.allSatisfy({ $0.isLetter }) {
                        body = String(body[body.index(after: newline)...])
                    }
                }

                let blockCalls = toolCalls(fromJSONText: body)
                if blockCalls.isEmpty || !blockCalls.allSatisfy({ knownToolNames.contains($0.name) }) {
                    // Not a tool call: keep the fenced block as visible text.
                    visibleText += text[index ..< closeRange.upperBound]
                } else {
                    visibleText += text[index ..< openRange.lowerBound]
                    calls.append(contentsOf: blockCalls)
                }
                index = closeRange.upperBound
            }

            guard !calls.isEmpty else { return nil }

            visibleText += text[index ..< text.endIndex]
            return ParseResult(
                visibleText: visibleText.trimmingCharacters(in: .whitespacesAndNewlines),
                calls: calls
            )
        }

        private static func parseBareJSON(
            _ text: String,
            knownToolNames: Set<String>
        ) -> ParseResult? {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.hasPrefix("{") || trimmed.hasPrefix("[") else { return nil }

            let calls = toolCalls(fromJSONText: trimmed)
            guard !calls.isEmpty, calls.allSatisfy({ knownToolNames.contains($0.name) }) else {
                return nil
            }
            return ParseResult(visibleText: "", calls: calls)
        }

        // MARK: JSON shaping

        private static func toolCalls(fromJSONText text: String) -> [ParsedToolCall] {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data)
            else { return [] }

            if let array = json as? [Any] {
                return array.compactMap { element in
                    (element as? [String: Any]).flatMap { toolCall(from: $0) }
                }
            }
            if let object = json as? [String: Any], let call = toolCall(from: object) {
                return [call]
            }
            return []
        }

        private static func toolCall(from object: [String: Any]) -> ParsedToolCall? {
            // Unwrap the OpenAI-style `{"type": "function", "function": {…}}` envelope.
            let source = (object["function"] as? [String: Any]) ?? object

            guard let name = source["name"] as? String, !name.isEmpty else { return nil }

            var argumentsValue = source["arguments"] ?? source["parameters"] ?? source["args"]
            // Some templates emit the arguments as a JSON-encoded string.
            if let argumentsString = argumentsValue as? String {
                argumentsValue =
                    argumentsString.data(using: .utf8)
                    .flatMap { try? JSONSerialization.jsonObject(with: $0) }
            }
            let argumentsObject = (argumentsValue as? [String: Any]) ?? [:]

            guard
                let data = try? JSONSerialization.data(
                    withJSONObject: argumentsObject,
                    options: [.sortedKeys]
                ),
                let argumentsJSON = String(data: data, encoding: .utf8)
            else { return nil }

            return ParsedToolCall(
                name: name,
                argumentsJSON: argumentsJSON,
                arguments: sendableJSONObject(from: argumentsObject)
            )
        }

        /// Re-types a `JSONSerialization` object graph as `Sendable` so it can be handed to the
        /// tokenizer's chat template.
        static func sendableJSONObject(from object: [String: Any]) -> [String: any Sendable] {
            var converted: [String: any Sendable] = [:]
            converted.reserveCapacity(object.count)
            for (key, value) in object {
                converted[key] = sendableJSONValue(from: value)
            }
            return converted
        }

        private static func sendableJSONValue(from value: Any) -> any Sendable {
            if let string = value as? String { return string }
            if let array = value as? [Any] { return array.map { sendableJSONValue(from: $0) } }
            if let dictionary = value as? [String: Any] { return sendableJSONObject(from: dictionary) }
            if value is NSNull {
                // Erased `nil`, which the template engine degrades to null.
                let null: String? = nil
                return null as any Sendable
            }
            if let number = value as? NSNumber {
                // `as? Bool` is not reliable here: JSON booleans and the integers 0 and 1 both
                // bridge to NSNumber, so distinguish them by the underlying CoreFoundation type.
                if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue }
                if let integer = Int(exactly: number) { return integer }
                return number.doubleValue
            }
            return String(describing: value)
        }
    }
#endif  // CoreML
