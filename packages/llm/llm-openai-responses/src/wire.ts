/**
 * Wire vocabulary and request assembly for the pure OpenAI Responses API.
 *
 * The request body is deliberately MINIMAL: only fields every Responses
 * endpoint understands (model, input, instructions, tools, reasoning,
 * temperature, max_output_tokens, store, stream). Fields from the newest
 * OpenAI surface — `prompt_cache_retention`, `prompt_cache_options`,
 * `service_tier` — are never emitted: gateways in front of volc/minimax/kimi
 * 400 the whole request on them ("Invalid request Error"), which is exactly
 * what this adapter exists to avoid. `stop` has no Responses equivalent and
 * is silently ignored.
 *
 * @module dsh-llm-openai-responses/wire
 */

import { CallId, LlmError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, Message, ToolSchema } from '@deepseek-ai/dsh-llm'

/** One input item on the Responses wire. Only shapes this adapter emits. */
export type ResponsesInputItem =
  | { type: 'message'; role: 'user'; content: { type: 'input_text'; text: string }[] }
  | { type: 'message'; role: 'assistant'; content: { type: 'output_text'; text: string }[] }
  | { type: 'function_call'; name: string; call_id: string; arguments: string }
  | { type: 'function_call_output'; call_id: string; output: string }

/** Body for `POST {baseURL}/responses`. */
export interface ResponsesRequestBody {
  model: string
  input: ResponsesInputItem[]
  instructions?: string
  tools?: { type: 'function'; name: string; description: string; parameters: Record<string, unknown>; strict: boolean }[]
  reasoning?: { effort: string; summary: 'auto' }
  temperature?: number
  max_output_tokens?: number
  store: false
  stream: true
}

/** Reasoning effort as spoken on the Responses wire. */
export type WireEffort = 'low' | 'medium' | 'high'

/** Map a harness reasoning effort onto the Responses wire; `off` disables. */
export function wireEffortOf(effort: string | undefined): WireEffort | undefined {
  switch (effort) {
    case undefined:
    case '':
      return 'high'
    case 'off':
      return undefined
    case 'low':
      return 'low'
    case 'medium':
      return 'medium'
    default:
      // 'high', 'max', and any merge-extensible newcomer clamp to high.
      return 'high'
  }
}

function flattenText(blocks: readonly ContentBlock[]): string {
  return blocks.map(block => block.type === 'text' ? block.text : '').join('')
}

function flattenToolResultText(blocks: readonly ContentBlock[]): string {
  return blocks.map((block) => {
    if (block.type === 'text') return block.text
    if (block.type === 'tool-result') return flattenToolResultText(block.content)
    return ''
  }).join('')
}

function requireTextOnly(message: Message): void {
  for (const block of message.content) {
    if (block.type === 'image') {
      throw new LlmError('llm-openai-responses: this route is text-only', 'UNSUPPORTED_CONTENT')
    }
  }
}

/**
 * Assemble the Responses request body from a harness call.
 * @param options - the fully-assembled harness request.
 * @param thinking - profile thinking policy; `disabled` strips reasoning from the wire.
 * @returns the body value for `POST {baseURL}/responses`.
 */
export function buildResponsesRequest(options: GenerateOptions, thinking: 'enabled' | 'disabled' | undefined): ResponsesRequestBody {
  const input: ResponsesInputItem[] = []
  let pendingSystemText = ''

  const flushPendingSystem = (): void => {
    if (pendingSystemText.length > 0) {
      input.push({ type: 'message', role: 'user', content: [{ type: 'input_text', text: pendingSystemText }] })
      pendingSystemText = ''
    }
  }

  for (const message of options.messages) {
    requireTextOnly(message)
    if (message.role === 'system') {
      // The Responses envelope has one instructions slot (options.system);
      // rare in-history system messages fold into the nearest user turn to
      // preserve ordering, the same degradation pi-ai's converter makes.
      pendingSystemText += (pendingSystemText.length > 0 ? '\n\n' : '') + flattenText(message.content)
      continue
    }
    if (message.role === 'assistant') {
      flushPendingSystem()
      const text = flattenText(message.content)
      if (text.length > 0) {
        input.push({ type: 'message', role: 'assistant', content: [{ type: 'output_text', text }] })
      }
      for (const block of message.content) {
        if (block.type === 'tool-call') {
          input.push({
            type: 'function_call',
            name: block.name,
            call_id: block.id,
            arguments: block.arguments,
          })
        }
      }
      continue
    }
    // user role: text first, then each tool result as its own output item.
    const text = flattenText(message.content)
    if (text.length > 0) {
      flushPendingSystem()
      input.push({ type: 'message', role: 'user', content: [{ type: 'input_text', text }] })
    }
    for (const block of message.content) {
      if (block.type !== 'tool-result') continue
      flushPendingSystem()
      input.push({
        type: 'function_call_output',
        call_id: block.toolCallId,
        output: flattenToolResultText(block.content) || '(no output)',
      })
    }
  }
  flushPendingSystem()

  const body: ResponsesRequestBody = {
    model: options.model,
    input,
    ...options.system !== undefined && options.system.length > 0 ? { instructions: options.system } : {},
    store: false,
    stream: true,
  }
  if (options.tools !== undefined && options.tools.length > 0) {
    body.tools = options.tools.map((tool: ToolSchema) => ({
      type: 'function',
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
      strict: false,
    }))
  }
  if (thinking !== 'disabled') {
    const effort = wireEffortOf(options.reasoningEffort)
    if (effort !== undefined) body.reasoning = { effort, summary: 'auto' }
  }
  if (options.temperature !== undefined) body.temperature = options.temperature
  if (options.maxTokens !== undefined) body.max_output_tokens = options.maxTokens
  return body
}

export { CallId }
