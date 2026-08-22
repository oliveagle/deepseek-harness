/**
 * Pure OpenAI chat-completions wire format. Types only.
 *
 * The one deliberate divergence from the DeepSeek wire shape (and the reason
 * this adapter exists): assistant history `tool_calls[].function.arguments`
 * travels as a JSON OBJECT, not a JSON-encoded string. Endpoints that parse
 * the request body straight into typed tool-call records (sglang, vLLM
 * function-calling validators, one-api strict mode) accept the object form
 * without a re-parse and reject string-encoded non-objects on some paths;
 * clients that replay history verbatim then brick every later turn.
 *
 * @module dsh-llm-openai/types
 */

/** Request body for `POST {baseURL}/chat/completions`. */
export interface WireRequest {
  model: string
  messages: WireMessage[]
  stream: true
  stream_options: { include_usage: true }
  tools?: WireTool[]
  temperature?: number
  max_tokens?: number
  /** Stop sequences; generation halts on the first one produced. */
  stop?: string[]
  /**
   * Gateway-specific extras (`requestExtras` plugin config) ride along under
   * their own keys, e.g. sglang's `chat_template_kwargs`. Core request fields
   * always win over an extras key of the same name.
   */
  [key: string]: unknown
}

/** System-role message: a single string of instructions. */
export interface WireSystemMessage {
  role: 'system'
  content: string
}

/** User-role message: a single string of user input. */
export interface WireUserMessage {
  role: 'user'
  content: string
}

/** Tool-role message: the result of one tool call, keyed by its call id. */
export interface WireToolMessage {
  role: 'tool'
  tool_call_id: string
  content: string
}

/** Assistant-role history message. `content` is always a string (never null). */
export interface WireAssistantMessage {
  role: 'assistant'
  content: string
  /** Completed calls; `arguments` is a plain JSON object. */
  tool_calls?: WireToolCall[]
}

/** One entry of the request `messages` array, discriminated on `role`. */
export type WireMessage =
  | WireSystemMessage
  | WireUserMessage
  | WireAssistantMessage
  | WireToolMessage

/** A completed tool call replayed on an assistant history message. */
export interface WireToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: Record<string, unknown> }
}

/** One entry of the request `tools` array; `parameters` is a JSON Schema object. */
export interface WireTool {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
  }
}
