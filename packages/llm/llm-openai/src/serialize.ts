/**
 * Serialize harness messages into pure OpenAI chat completions. Message
 * structure (system/user/assistant text, tool-result expansion, tool-call
 * identity self-healing) is delegated to the DeepSeek wire serializer — the
 * two routes share the exact OpenAI message grammar — and this module then
 * applies the one pure-OpenAI divergence: every assistant history
 * `tool_calls[].function.arguments` is emitted as a JSON OBJECT (parsed,
 * recursively unwrapped from quoted encodings, `{}` fallback), and no
 * DeepSeek-specific fields (`reasoning_content`, `thinking`,
 * `reasoning_effort`) ever reach the wire. Gateway extras
 * (`chat_template_kwargs` and friends) merge under their own keys with core
 * request fields winning collisions.
 * @module dsh-llm-openai/serialize
 */

import type { GenerateOptions } from '@deepseek-ai/dsh-llm'
// Cross-package reuse goes through the package's deliberate './src/*'
// subpath export: the DeepSeek root deliberately keeps wire helpers off its
// public surface (guarded by its adapter.spec 'keeps wire helpers off the
// package root').
import { serializeMessages } from '@deepseek-ai/dsh-llm-deepseek/src/serialize.ts'
import type { WireMessage as DeepSeekWireMessage } from '@deepseek-ai/dsh-llm-deepseek/src/types.ts'
import type { WireMessage, WireRequest, WireTool } from './types.ts'

/** Bound on recursive unwrapping of quoted JSON strings in tool arguments. */
const MAX_UNWRAP_DEPTH = 8

/**
 * Parse one tool-call arguments value into the OpenAI object form. A plain
 * JSON object string parses directly; a quoted JSON string (a model under
 * speculative decoding sometimes double-encodes) is unwrapped recursively;
 * anything else degrades to `{}` — the history stays replayable instead of
 * letting one malformed argument from an already-executed call poison every
 * later turn of the session.
 */
export function objectArguments(args: string): Record<string, unknown> {
  let value: unknown = args
  for (let depth = 0; depth < MAX_UNWRAP_DEPTH; depth += 1) {
    if (typeof value !== 'string') break
    try {
      value = JSON.parse(value)
    } catch {
      return {}
    }
  }
  return isPlainObject(value) ? value : {}
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Convert DeepSeek-serialized wire messages to the pure OpenAI shape:
 * strip `reasoning_content` and rewrite `arguments` strings to objects.
 * Message order, tool-call ids, and text content are preserved verbatim.
 */
export function toObjectStyle(messages: DeepSeekWireMessage[]): WireMessage[] {
  return messages.map((message) => {
    if (message.role !== 'assistant') return message
    // The DeepSeek builder types content as string|null but only ever emits
    // "" on text-less turns, so the narrowing assertion adds no runtime path.
    const content = message.content as string
    return {
      role: 'assistant' as const,
      content,
      // reasoning_content is DeepSeek-specific and never travels on this route.
      ...message.tool_calls === undefined ? {} : {
        tool_calls: message.tool_calls.map(call => ({
          id: call.id,
          type: 'function' as const,
          function: { name: call.function.name, arguments: objectArguments(call.function.arguments) },
        })),
      },
    }
  })
}

/**
 * Build the full wire request. Always streaming (usage reporting on);
 * optional fields are omitted rather than sent as null, so endpoint defaults
 * apply. `extras` (gateway-specific request fields) merge first — a core
 * field always wins a same-key collision.
 * @param options - the harness request (model, history, system, tools, sampling).
 * @param extras - verbatim extra body fields from the plugin config, if any.
 * @returns the chat-completions request body.
 */
export function serializeRequest(
  options: GenerateOptions,
  extras?: Record<string, unknown>,
): WireRequest {
  const messages: WireMessage[] = options.system === undefined
    ? []
    : [{ role: 'system', content: options.system }]
  messages.push(...toObjectStyle(serializeMessages(options.messages)))

  const tools: WireTool[] | undefined = options.tools?.map(tool => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }))

  return {
    ...extras,
    model: options.model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    ...tools !== undefined && tools.length > 0 ? { tools } : {},
    ...options.temperature !== undefined ? { temperature: options.temperature } : {},
    ...options.maxTokens === undefined ? {} : { max_tokens: options.maxTokens },
    ...options.stop !== undefined ? { stop: options.stop } : {},
  }
}
