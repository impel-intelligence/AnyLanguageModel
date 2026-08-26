import Foundation
import Testing

@testable import AnyLanguageModel

#if CoreML
    import Hub
    import CoreML

    private let shouldRunCoreMLTests: Bool = {
        // Enable when explicitly requested via environment variable
        if ProcessInfo.processInfo.environment["ENABLE_COREML_TESTS"] != nil {
            return true
        }

        // Skip in CI environments
        if ProcessInfo.processInfo.environment["CI"] != nil {
            return false
        }

        return true
    }()

    @Suite("CoreMLLanguageModel", .enabled(if: shouldRunCoreMLTests), .serialized)
    struct CoreMLLanguageModelTests {
        let modelId = "apple/mistral-coreml"
        let modelPackageName = "StatefulMistral7BInstructInt4.mlpackage"

        @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        private static let modelTask = Task {
            let hasToken = ProcessInfo.processInfo.environment["HF_TOKEN"] != nil
            let hubApi = HubApi(useOfflineMode: !hasToken)
            let repoURL = try await hubApi.snapshot(
                from: Hub.Repo(id: "apple/mistral-coreml", type: .models),
                matching: "*Int4.mlpackage/**"
            ) { progress in
                print("Download progress: \(Int(progress.fractionCompleted * 100))%")
            }

            let modelURL = repoURL.appending(component: "StatefulMistral7BInstructInt4.mlpackage")
            let compiledURL: URL
            if modelURL.pathExtension == "mlmodelc" {
                compiledURL = modelURL
            } else {
                compiledURL = try await MLModel.compileModel(at: modelURL)
            }
            return try await CoreMLLanguageModel(url: compiledURL)
        }

        @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func getModel() async throws -> CoreMLLanguageModel {
            try await Self.modelTask.value
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func basicResponse() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say hello")
            #expect(!response.content.isEmpty)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func withGenerationOptions() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 32
            )

            let response = try await session.respond(
                to: "Tell me a fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func withSamplingStrategies() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            // Test greedy sampling
            let greedyOptions = GenerationOptions(sampling: .greedy)
            let greedyResponse = try await session.respond(
                to: "Complete this sentence: The sky is",
                options: greedyOptions
            )
            #expect(!greedyResponse.content.isEmpty)

            // Test top-k sampling
            let topKOptions = GenerationOptions(sampling: .random(top: 10))
            let topKResponse = try await session.respond(
                to: "Complete this sentence: The sky is",
                options: topKOptions
            )
            #expect(!topKResponse.content.isEmpty)

            // Test nucleus sampling
            let nucleusOptions = GenerationOptions(sampling: .random(probabilityThreshold: 0.9))
            let nucleusResponse = try await session.respond(
                to: "Complete this sentence: The sky is",
                options: nucleusOptions
            )
            #expect(!nucleusResponse.content.isEmpty)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func temperatureVariations() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            // Test low temperature (more deterministic)
            let lowTempOptions = GenerationOptions(temperature: 0.1)
            let lowTempResponse = try await session.respond(
                to: "Write a short story about a cat",
                options: lowTempOptions
            )
            #expect(!lowTempResponse.content.isEmpty)

            // Test high temperature (more creative)
            let highTempOptions = GenerationOptions(temperature: 0.9)
            let highTempResponse = try await session.respond(
                to: "Write a short story about a cat",
                options: highTempOptions
            )
            #expect(!highTempResponse.content.isEmpty)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func maxTokensLimit() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(maximumResponseTokens: 5)
            let response = try await session.respond(
                to: "Write a long story about space exploration",
                options: options
            )
            #expect(!response.content.isEmpty)
            // Note: We can't easily test token count without access to the tokenizer
            // but we can verify the response is not empty
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func multimodal_rejectsImageURL() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)
            do {
                _ = try await session.respond(
                    to: "Describe this image",
                    image: .init(url: testImageURL)
                )
                Issue.record("Expected error when image segments are present")
            } catch {
                // CoreMLUnsupportedFeatureError is a private struct, so we just check that an error is thrown
                #expect(Bool(true))
            }
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func multimodal_rejectsImageData() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)
            do {
                _ = try await session.respond(
                    to: "Describe this image",
                    image: .init(data: testImageData, mimeType: "image/jpeg")
                )
                Issue.record("Expected error when image segments are present")
            } catch {
                // CoreMLUnsupportedFeatureError is a private struct, so we just check that an error is thrown
                #expect(Bool(true))
            }
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationSimpleString() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a greeting message that says hello",
                generating: SimpleString.self
            )
            #expect(!response.content.message.isEmpty)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationSimpleInt() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a count value of 42",
                generating: SimpleInt.self
            )
            #expect(response.content.count >= 0)
            let jsonData = response.rawContent.jsonString.data(using: .utf8)
            #expect(jsonData != nil)
            if let jsonData {
                let json = try JSONSerialization.jsonObject(with: jsonData)
                let dictionary = json as? [String: Any]
                #expect(dictionary != nil)
                if let dictionary {
                    let countValue = dictionary["count"] as? NSNumber
                    #expect(countValue?.intValue != nil)
                }
            }
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationSimpleBool() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a boolean value: true",
                generating: SimpleBool.self
            )
            let jsonData = response.rawContent.jsonString.data(using: .utf8)
            #expect(jsonData != nil)
            if let jsonData {
                let json = try JSONSerialization.jsonObject(with: jsonData)
                let dictionary = json as? [String: Any]
                #expect(dictionary != nil)
                if let dictionary {
                    let boolValue = dictionary["value"] as? Bool
                    #expect(boolValue != nil)
                }
            }
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationSimpleDouble() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a temperature value of 72.5 degrees",
                generating: SimpleDouble.self
            )
            #expect(!response.content.temperature.isNaN)
            #expect(response.content.temperature.isFinite)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationOptionalFields() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a person named Alex with nickname 'Lex'. Nickname may be omitted if unsure.",
                generating: OptionalFields.self
            )
            #expect(!response.content.name.isEmpty)
            if let nickname = response.content.nickname {
                #expect(!nickname.isEmpty)
            }
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationEnum() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a high priority value",
                generating: Priority.self
            )
            #expect([Priority.low, Priority.medium, Priority.high].contains(response.content))
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationSimpleArray() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a list of 3 color names: red, green, blue",
                generating: SimpleArray.self
            )
            #expect(!response.content.colors.isEmpty)
        }

        /// Requires a model whose chat template supports tools. The template renders the tool specs,
        /// the model answers with tool-call text, and the parser turns that back into transcript
        /// entries.
        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func withTools() async throws {
            let model = try await getModel()
            let weatherTool = WeatherTool()
            let session = LanguageModelSession(model: model, tools: [weatherTool])

            let response = try await session.respond(to: "How's the weather in San Francisco?")

            var foundToolOutput = false
            for case let .toolOutput(toolOutput) in response.transcriptEntries {
                #expect(!toolOutput.id.isEmpty)
                #expect(toolOutput.toolName == "getWeather")
                foundToolOutput = true
            }
            #expect(foundToolOutput)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func streamWithTools() async throws {
            let model = try await getModel()
            let weatherTool = WeatherTool()
            let session = LanguageModelSession(model: model, tools: [weatherTool])

            let stream = session.streamResponse(to: "How's the weather in San Francisco?")

            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []

            var toolAppearedInTranscript: Bool = false
            var toolResponseAppearedInTranscript: Bool = false

            for try await snapshot in stream {
                snapshots.append(snapshot)

                for entry in session.transcript {
                    switch entry {
                    case .toolCalls:
                        toolAppearedInTranscript = true
                    case .toolOutput:
                        toolResponseAppearedInTranscript = true
                    default: break
                    }
                }
            }

            #expect(toolAppearedInTranscript, "Expected a tool call to appear in the transcript during streaming.")
            #expect(
                toolResponseAppearedInTranscript,
                "Expected a tool output to appear in the transcript during streaming."
            )

            // Tool-call markup must never be published as assistant text.
            if #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, watchOS 26.0, *) {
                for snapshot in snapshots {
                    #expect(!snapshot.content.contains("<tool_call>"))
                    #expect(!snapshot.content.contains("[TOOL_CALLS]"))
                }
            }
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationNestedStruct() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a person named John, age 25, living at 123 Main St, Springfield, 12345",
                generating: StructuredPerson.self
            )
            #expect(!response.content.name.isEmpty)
            #expect(response.content.age >= 0)
            #expect(!response.content.address.street.isEmpty)
            #expect(!response.content.address.city.isEmpty)
        }
    }

    /// Exercises the tool-call text formats the Core ML provider claims to support. These need no
    /// downloaded model, so unlike the suite above they run everywhere the CoreML trait is enabled.
    @Suite("CoreMLToolCallParsing")
    struct CoreMLToolCallParsingTests {
        private let knownToolNames: Set<String> = ["getWeather"]

        @Test func parsesHermesStyleTaggedCall() {
            let result = CoreMLToolCallParser.parse(
                "Let me look that up.\n<tool_call>\n{\"name\": \"getWeather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
                knownToolNames: knownToolNames
            )

            #expect(result.visibleText == "Let me look that up.")
            #expect(result.calls.count == 1)
            #expect(result.calls.first?.name == "getWeather")
            #expect(result.calls.first?.argumentsJSON == "{\"city\":\"Paris\"}")
        }

        @Test func parsesMultipleTaggedCalls() {
            let result = CoreMLToolCallParser.parse(
                "<tool_call>{\"name\": \"getWeather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>"
                    + "<tool_call>{\"name\": \"getWeather\", \"arguments\": {\"city\": \"Oslo\"}}</tool_call>",
                knownToolNames: knownToolNames
            )

            #expect(result.calls.count == 2)
            #expect(result.visibleText.isEmpty)
        }

        @Test func parsesMistralToolCallsMarker() {
            let result = CoreMLToolCallParser.parse(
                "[TOOL_CALLS] [{\"name\": \"getWeather\", \"arguments\": {\"city\": \"Paris\"}}]",
                knownToolNames: knownToolNames
            )

            #expect(result.calls.count == 1)
            #expect(result.calls.first?.name == "getWeather")
            #expect(result.visibleText.isEmpty)
        }

        @Test func parsesLlamaPythonTagWithParametersKey() {
            let result = CoreMLToolCallParser.parse(
                "<|python_tag|>{\"name\": \"getWeather\", \"parameters\": {\"city\": \"Paris\"}}",
                knownToolNames: knownToolNames
            )

            #expect(result.calls.count == 1)
            #expect(result.calls.first?.argumentsJSON == "{\"city\":\"Paris\"}")
        }

        @Test func parsesArgumentsEncodedAsJSONString() {
            let result = CoreMLToolCallParser.parse(
                "<tool_call>{\"name\": \"getWeather\", \"arguments\": \"{\\\"city\\\": \\\"Paris\\\"}\"}</tool_call>",
                knownToolNames: knownToolNames
            )

            #expect(result.calls.first?.argumentsJSON == "{\"city\":\"Paris\"}")
        }

        @Test func parsesOpenAIFunctionEnvelope() {
            let result = CoreMLToolCallParser.parse(
                "<tool_call>{\"type\": \"function\", \"function\": {\"name\": \"getWeather\", \"arguments\": {\"city\": \"Paris\"}}}</tool_call>",
                knownToolNames: knownToolNames
            )

            #expect(result.calls.first?.name == "getWeather")
        }

        @Test func parsesBareJSONOnlyForKnownTools() {
            let text = "{\"name\": \"getWeather\", \"arguments\": {\"city\": \"Paris\"}}"

            let known = CoreMLToolCallParser.parse(text, knownToolNames: knownToolNames)
            #expect(known.calls.count == 1)

            let unknown = CoreMLToolCallParser.parse(text, knownToolNames: [])
            #expect(unknown.calls.isEmpty)
            #expect(unknown.visibleText == text)
        }

        @Test func treatsPlainProseAsText() {
            let result = CoreMLToolCallParser.parse(
                "The weather in Paris is sunny.",
                knownToolNames: knownToolNames
            )

            #expect(result.calls.isEmpty)
            #expect(result.visibleText == "The weather in Paris is sunny.")
        }

        @Test func treatsUnparseableTagBodyAsText() {
            let text = "<tool_call>not json</tool_call>"
            let result = CoreMLToolCallParser.parse(text, knownToolNames: knownToolNames)

            #expect(result.calls.isEmpty)
            #expect(result.visibleText == text)
        }

        @Test func withholdsToolCallMarkupWhileStreaming() {
            #expect(
                CoreMLToolCallParser.visibleTextForStreaming("Checking. <tool_call>{\"na")
                    == "Checking. "
            )
            #expect(CoreMLToolCallParser.visibleTextForStreaming("[TOOL_CALLS] [{\"na").isEmpty)
            #expect(CoreMLToolCallParser.visibleTextForStreaming("{\"name\": \"get").isEmpty)
            #expect(
                CoreMLToolCallParser.visibleTextForStreaming("The weather is") == "The weather is"
            )
        }

        @Test func signatureDistinguishesArgumentsAndIgnoresKeyOrder() {
            let first = CoreMLToolCallParser.parse(
                "<tool_call>{\"name\": \"getWeather\", \"arguments\": {\"city\": \"Paris\", \"unit\": \"C\"}}</tool_call>",
                knownToolNames: knownToolNames
            )
            let reordered = CoreMLToolCallParser.parse(
                "<tool_call>{\"name\": \"getWeather\", \"arguments\": {\"unit\": \"C\", \"city\": \"Paris\"}}</tool_call>",
                knownToolNames: knownToolNames
            )
            let different = CoreMLToolCallParser.parse(
                "<tool_call>{\"name\": \"getWeather\", \"arguments\": {\"city\": \"Oslo\"}}</tool_call>",
                knownToolNames: knownToolNames
            )

            #expect(
                CoreMLToolCallParser.signature(for: first.calls)
                    == CoreMLToolCallParser.signature(for: reordered.calls)
            )
            #expect(
                CoreMLToolCallParser.signature(for: first.calls)
                    != CoreMLToolCallParser.signature(for: different.calls)
            )
        }

        @Test func toolCallIDsAreNineAlphanumericCharacters() {
            // Mistral's chat template rejects anything else.
            for _ in 0 ..< 32 {
                let id = CoreMLToolCallParser.makeToolCallID()
                #expect(id.count == 9)
                #expect(id.allSatisfy { $0.isLetter || $0.isNumber })
            }
        }
    }
#endif  // CoreML
