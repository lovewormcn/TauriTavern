use serde_json::{Map, Value};

use crate::errors::ApplicationError;
use tt_ports::repositories::chat_completion_repository::ChatCompletionSource;

use super::OPENCODE_STABLE_CHAT_ID_FIELD;
use super::exchange::ChatCompletionProviderFormat;
use super::opencode::{self, OpenCodeApiFormat};

mod aws_bedrock;
mod chutes;
mod claude;
mod claude_messages;
mod cohere;
mod content_parts;
mod custom;
mod deepseek;
mod gemini_interactions;
mod makersuite;
mod minimax;
mod moonshot;
mod nanogpt;
mod openai;
mod openai_reasoning;
mod openai_responses;
mod openrouter;
mod prompt_post_processing;
mod shared;
mod tool_calls;
mod tool_choice;
mod vertexai;
mod workers_ai;
mod xai;
mod zai;

pub(super) fn build_payload(
    source: ChatCompletionSource,
    payload: Map<String, Value>,
) -> Result<(String, Value), ApplicationError> {
    let mut payload = payload;
    let opencode_format = (source == ChatCompletionSource::OpenCode)
        .then(|| opencode::format_from_payload(&payload))
        .transpose()?;
    payload.remove(OPENCODE_STABLE_CHAT_ID_FIELD);
    if opencode_format.is_some() {
        payload.remove("opencode_endpoint");
        payload.remove("opencode_api_format");
    }

    if !matches!(source, ChatCompletionSource::DeepSeek) {
        prompt_post_processing::apply_custom_prompt_post_processing(&mut payload);
    }

    if source == ChatCompletionSource::OpenAi
        && ChatCompletionProviderFormat::from_payload(source, &payload)?
            == ChatCompletionProviderFormat::OpenAiResponses
    {
        return openai_responses::build(payload);
    }

    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let has_tools = payload
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());

    let (endpoint, mut upstream_payload) = match source {
        ChatCompletionSource::OpenAi
        | ChatCompletionSource::Groq
        | ChatCompletionSource::SiliconFlow
        | ChatCompletionSource::Pollinations => openai::build(payload),
        ChatCompletionSource::OpenCode => {
            match opencode_format.expect("OpenCode format resolved") {
                OpenCodeApiFormat::OpenAiCompat => openai::build_chat(payload),
                OpenCodeApiFormat::OpenAiResponses => openai_responses::build(payload),
                OpenCodeApiFormat::ClaudeMessages => claude_messages::build(payload),
                OpenCodeApiFormat::Gemini => makersuite::build(payload),
            }
        }
        ChatCompletionSource::DeepSeek => deepseek::build(payload),
        ChatCompletionSource::Cohere => Ok(cohere::build(payload)?),
        ChatCompletionSource::Moonshot => moonshot::build(payload),
        ChatCompletionSource::NanoGpt => nanogpt::build(payload),
        ChatCompletionSource::Chutes => chutes::build(payload),
        ChatCompletionSource::Xai => xai::build(payload),
        ChatCompletionSource::WorkersAi => workers_ai::build(payload),
        ChatCompletionSource::OpenRouter => openrouter::build(payload),
        ChatCompletionSource::Zai => zai::build(payload),
        ChatCompletionSource::MiniMax => Ok(minimax::build(payload)),
        ChatCompletionSource::Custom => custom::build(payload),
        ChatCompletionSource::Claude => Ok(claude::build(payload)?),
        ChatCompletionSource::AwsBedrock => Ok(aws_bedrock::build(payload)?),
        ChatCompletionSource::Makersuite => Ok(makersuite::build(payload)?),
        ChatCompletionSource::VertexAi => Ok(vertexai::build(payload)?),
    }?;

    // DeepSeek V4 thinking models require `reasoning_content` on every
    // assistant message in a tool context. OpenAI-compatible sources that
    // only see the model name (custom / openrouter / opencode) reuse the
    // same DeepSeek fix-up so tool follow-ups do not fail with 400.
    if source != ChatCompletionSource::DeepSeek
        && deepseek::is_deepseek_v4_model(&model)
        && endpoint == "/chat/completions"
        && let Some(body) = upstream_payload.as_object_mut()
        && let Some(messages) = body.get_mut("messages")
        && let Some(messages) = messages.as_array_mut()
    {
        deepseek::ensure_tool_context_reasoning_content(messages, has_tools)?;
    }

    Ok((endpoint, upstream_payload))
}

pub(super) fn validate_upstream_tool_transcript(
    endpoint_path: &str,
    upstream_payload: &Value,
) -> Result<(), ApplicationError> {
    if endpoint_path != "/chat/completions" {
        return Ok(());
    }

    tool_calls::validate_openai_chat_tool_transcript(upstream_payload.get("messages"), false)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::build_payload;
    use tt_ports::repositories::chat_completion_repository::ChatCompletionSource;

    #[test]
    fn converted_tool_call_history_replays_unusable_arguments_as_an_empty_object() {
        let error = "## Tool error\n\nRequest rejected";
        for arguments in ["null", "[1,2]", r#"{"path":"#] {
            for (format, arguments_path, error_path) in [
                ("openai_responses", "/input/1/arguments", "/input/2/output"),
                (
                    "claude_messages",
                    "/messages/1/content/0/input",
                    "/messages/2/content/0/content",
                ),
                (
                    "gemini_generate_content",
                    "/contents/1/parts/0/functionCall/args",
                    "/contents/2/parts/0/functionResponse/response/content",
                ),
                (
                    "gemini_interactions",
                    "/input/1/arguments",
                    "/input/2/result/0/text",
                ),
            ] {
                let payload = json!({
                    "chat_completion_source": "custom",
                    "custom_api_format": format,
                    "model": "test-model",
                    "tools": [{ "type": "function", "function": {
                        "name": "read_file", "parameters": { "type": "object" }
                    } }],
                    "messages": [
                        { "role": "user", "content": "Read a file" },
                        { "role": "assistant", "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": { "name": "read_file", "arguments": arguments }
                        }] },
                        { "role": "tool", "tool_call_id": "call_1", "content": error }
                    ]
                });
                let (_, upstream) = build_payload(
                    ChatCompletionSource::Custom,
                    payload.as_object().unwrap().clone(),
                )
                .unwrap();
                let replay = upstream
                    .pointer(arguments_path)
                    .unwrap_or_else(|| panic!("{format} missing {arguments_path}: {upstream}"));
                let replay = match replay.as_str() {
                    Some(encoded) => serde_json::from_str::<Value>(encoded).unwrap(),
                    None => replay.clone(),
                };
                assert_eq!(replay, json!({}), "{format}: {arguments}");
                assert_eq!(
                    upstream.pointer(error_path),
                    Some(&json!(error)),
                    "{format}"
                );
            }
        }
    }

    #[test]
    fn claude_leaves_additional_body_overrides_to_service_layer() {
        let payload = json!({
            "chat_completion_source": "claude",
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "custom_include_body": "{\"metadata\":{\"feature\":\"override\"}}",
            "custom_exclude_body": "[\"stream\"]"
        })
        .as_object()
        .cloned()
        .expect("payload must be object");

        let (_, upstream) =
            build_payload(ChatCompletionSource::Claude, payload).expect("payload should build");
        let body = upstream.as_object().expect("body must be object");

        assert!(body.get("metadata").is_none());
        assert_eq!(
            body.get("stream").and_then(serde_json::Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn openai_gpt_6_astra_uses_responses_api() {
        let payload = json!({
            "model": "gpt-6-astra",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7
        })
        .as_object()
        .cloned()
        .expect("payload must be object");

        let (endpoint, upstream) =
            build_payload(ChatCompletionSource::OpenAi, payload).expect("payload should build");

        assert_eq!(endpoint, "/responses");
        assert_eq!(upstream["model"], "gpt-6-astra");
        assert_eq!(upstream["temperature"], 0.7);
    }

    #[test]
    fn opencode_selects_existing_wire_adapter_explicitly() {
        for (format, endpoint, model) in [
            (
                "openai_compat",
                "/chat/completions",
                "gpt-3.5-turbo-instruct",
            ),
            ("openai_responses", "/responses", "test-model"),
            ("claude_messages", "/messages", "test-model"),
            ("gemini", "/generateContent", "test-model"),
        ] {
            let payload = json!({
                "chat_completion_source": "opencode",
                "opencode_endpoint": "zen",
                "opencode_api_format": format,
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": false
            })
            .as_object()
            .cloned()
            .unwrap();

            assert_eq!(
                build_payload(ChatCompletionSource::OpenCode, payload)
                    .unwrap()
                    .0,
                endpoint
            );
        }
    }

    #[test]
    fn compatible_sources_fill_missing_reasoning_content_for_deepseek_v4_tool_context() {
        for source in [
            ChatCompletionSource::OpenRouter,
            ChatCompletionSource::Custom,
            ChatCompletionSource::OpenCode,
        ] {
            for model in [
                "deepseek/deepseek-v4-flash",
                "deepseek/deepseek-flash",
                "deepseek-flash-v4.1",
                "deepseek-v4-pro-0813",
                "opencodego/deepseek-flash",
                "newapi/openrouter/deepseek-v4.1-flash",
            ] {
                let payload = json!({
                    "chat_completion_source": "custom",
                    "custom_api_format": "openai_compat",
                    "opencode_api_format": "openai_compat",
                    "model": model,
                    "messages": [
                        {"role":"user","content":"weather"},
                        {"role":"assistant","content":"I'll check."},
                        {"role":"user","content":"ok"},
                        {
                            "role":"assistant",
                            "content":"",
                            "tool_calls":[{
                                "id":"call_1",
                                "type":"function",
                                "function":{"name":"weather","arguments":"{}"}
                            }]
                        },
                        {"role":"tool","tool_call_id":"call_1","content":"cloudy"}
                    ],
                    "tools": [{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]
                })
                .as_object()
                .cloned()
                .expect("payload must be object");

                let (endpoint, upstream) = build_payload(source, payload)
                    .unwrap_or_else(|error| panic!("{source:?} {model}: {error}"));
                assert_eq!(endpoint, "/chat/completions", "{source:?} {model}");

                let messages = upstream
                    .get("messages")
                    .and_then(Value::as_array)
                    .expect("messages must be array");

                for index in [1_usize, 3] {
                    let assistant = messages
                        .get(index)
                        .and_then(Value::as_object)
                        .unwrap_or_else(|| panic!("{source:?} {model}: message {index} missing"));
                    assert_eq!(
                        assistant.get("reasoning_content").and_then(Value::as_str),
                        Some(""),
                        "{source:?} {model}: assistant {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn compatible_sources_keep_deepseek_3_2_messages_untouched() {
        let payload = json!({
            "chat_completion_source": "custom",
            "custom_api_format": "openai_compat",
            "model": "deepseek-3.2",
            "messages": [
                {"role":"user","content":"weather"},
                {"role":"assistant","content":"I'll check."},
                {
                    "role":"assistant",
                    "content":"",
                    "tool_calls":[{
                        "id":"call_1",
                        "type":"function",
                        "function":{"name":"weather","arguments":"{}"}
                    }]
                },
                {"role":"tool","tool_call_id":"call_1","content":"cloudy"}
            ],
            "tools": [{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]
        })
        .as_object()
        .cloned()
        .expect("payload must be object");

        let (_, upstream) = build_payload(ChatCompletionSource::Custom, payload)
            .expect("payload should build");
        let messages = upstream
            .get("messages")
            .and_then(Value::as_array)
            .expect("messages must be array");

        for index in [1_usize, 2] {
            let assistant = messages
                .get(index)
                .and_then(Value::as_object)
                .expect("assistant must be object");
            assert!(
                assistant.get("reasoning_content").is_none(),
                "deepseek-3.2 must stay untouched at {index}"
            );
        }
    }

    #[test]
    fn compatible_sources_do_not_touch_non_deepseek_models() {
        let payload = json!({
            "chat_completion_source": "custom",
            "custom_api_format": "openai_compat",
            "model": "qwen3-max",
            "messages": [
                {"role":"user","content":"weather"},
                {
                    "role":"assistant",
                    "content":"",
                    "tool_calls":[{
                        "id":"call_1",
                        "type":"function",
                        "function":{"name":"weather","arguments":"{}"}
                    }]
                },
                {"role":"tool","tool_call_id":"call_1","content":"cloudy"}
            ],
            "tools": [{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]
        })
        .as_object()
        .cloned()
        .expect("payload must be object");

        let (_, upstream) = build_payload(ChatCompletionSource::Custom, payload)
            .expect("payload should build");
        let messages = upstream
            .get("messages")
            .and_then(Value::as_array)
            .expect("messages must be array");

        let assistant = messages
            .get(1)
            .and_then(Value::as_object)
            .expect("assistant must be object");
        assert!(assistant.get("reasoning_content").is_none());
    }

    #[test]
    fn custom_openai_responses_replays_native_function_call_through_payload_boundary() {
        let payload = json!({
            "chat_completion_source": "custom",
            "custom_api_format": "openai_responses",
            "model": "gpt-5",
            "messages": [
                { "role": "user", "content": "hi" },
                {
                    "role": "assistant",
                    "content": "",
                    "native": {
                        "openai_responses": {
                            "responseId": "resp_1",
                            "output": [{
                                "id": "fc_1",
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "workspace_write_file",
                                "arguments": "{\"path\":\"output/main.md\",\"content\":\"hi\"}"
                            }]
                        }
                    }
                },
                { "role": "tool", "tool_call_id": "call_1", "content": "ok" }
            ]
        })
        .as_object()
        .cloned()
        .expect("payload must be object");

        let (endpoint, upstream) =
            build_payload(ChatCompletionSource::Custom, payload).expect("payload should build");

        assert_eq!(endpoint, "/responses");
        let input = upstream
            .get("input")
            .and_then(Value::as_array)
            .expect("responses input should exist");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_1");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_1");
    }

    #[test]
    fn custom_openai_responses_rejects_orphan_tool_output_through_payload_boundary() {
        let payload = json!({
            "chat_completion_source": "custom",
            "custom_api_format": "openai_responses",
            "model": "gpt-5",
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "tool", "tool_call_id": "call_1", "content": "orphan" }
            ]
        })
        .as_object()
        .cloned()
        .expect("payload must be object");

        let error = build_payload(ChatCompletionSource::Custom, payload)
            .expect_err("orphan tool output must fail");

        assert!(
            error
                .to_string()
                .contains("without preceding function_call")
        );
    }
}
