/**
 * Responses SSE event stream → harness StreamChunk translation.
 *
 * The Responses wire has no `[DONE]` sentinel; the terminal events are
 * `response.completed` / `response.incomplete` / `response.failed`. An EOF
 * without one is truncation and fails the call (`STREAM_CLOSED`), mirroring
 * the chat-completions transport's guarantee that a stream that never
 * terminated cannot be trusted.
 *
 * @module dsh-llm-openai-responses/translate
 */

import { EventSourceParserStream } from 'eventsource-parser/stream'
import {
  CONTEXT_WINDOW_EXCEEDED_CODE,
  isContextWindowExceededError,
  isQuotaExceededError,
  LlmError,
  QUOTA_EXCEEDED_CODE,
} from '@deepseek-ai/dsh-llm'
import type { ContentBlock, StreamChunk, TokenUsage } from '@deepseek-ai/dsh-llm'
import { CallId } from './wire.ts'

/** One parsed SSE event: the `event:` name plus its raw `data:` payload. */
export interface ResponsesSseEvent {
  event: string | undefined
  data: string
}

interface ItemState {
  index: number
  kind: 'text' | 'reasoning' | 'tool-call'
  text: string
  arguments: string
  callId: CallId
}

/** Decode an SSE byte stream into `event:`/`data:` pairs. */
export async function* parseResponsesSse(
  stream: ReadableStream<BufferSource>,
  onComment?: (comment: string) => void,
): AsyncGenerator<ResponsesSseEvent> {
  const events = stream
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new EventSourceParserStream({ onComment }))
  for await (const { event, data } of events) {
    yield { event: event === undefined ? '' : event, data }
  }
}

/** Usage accounting as reported on the terminal event. */
export interface WireUsage {
  input_tokens?: number
  input_tokens_details?: { cached_tokens?: number }
  output_tokens?: number
  output_tokens_details?: { reasoning_tokens?: number }
}

function tokenUsageOf(usage: WireUsage | undefined): TokenUsage {
  const input = usage?.input_tokens ?? 0
  const cached = usage?.input_tokens_details?.cached_tokens ?? 0
  const reasoning = usage?.output_tokens_details?.reasoning_tokens ?? 0
  const result: TokenUsage = {
    // The contract counts input DISJOINT from cache hits, so cached tokens
    // are subtracted out of the provider's total prompt count.
    inputTokens: Math.max(0, input - cached),
    outputTokens: usage?.output_tokens ?? 0,
  }
  if (cached > 0) result.cacheReadTokens = cached
  if (reasoning > 0) result.reasoningTokens = reasoning
  return result
}

function wireErrorCode(code: string | undefined, message: string | undefined): string {
  const detail = [code, message].filter(Boolean).join(' ')
  if (isQuotaExceededError(detail)) return QUOTA_EXCEEDED_CODE
  if (detail.toLowerCase().includes('rate limit')) return 'RATE_LIMIT'
  if (isContextWindowExceededError(detail)) return CONTEXT_WINDOW_EXCEEDED_CODE
  if (code === 'invalid_request_error' || code === 'invalid_request') return 'INVALID_REQUEST'
  return code !== undefined && code.length > 0 ? code : 'SERVER'
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

/**
 * Translate a Responses SSE event stream into harness chunks. Emits usage
 * before the terminal finish and nothing after; returns on the terminal
 * event, throws on `response.failed` / `error` / truncation.
 * @param events - parsed SSE events in arrival order.
 * @returns harness stream chunks for one model call.
 */
export async function* translateResponses(events: AsyncIterable<ResponsesSseEvent>): AsyncGenerator<StreamChunk> {
  const items = new Map<string, ItemState>()
  let nextIndex = 0
  let sawToolCall = false

  const itemOf = (itemId: string, kindHint?: ItemState['kind']): ItemState | undefined => {
    const existing = items.get(itemId)
    if (existing !== undefined) return existing
    if (kindHint === undefined) return undefined
    // Some gateways start deltas before (or instead of) output_item.added;
    // synthesize the slot so index correlation never breaks.
    const state: ItemState = { index: nextIndex++, kind: kindHint, text: '', arguments: '', callId: CallId(itemId) }
    items.set(itemId, state)
    return state
  }

  for await (const { event, data } of events) {
    let payload: Record<string, unknown>
    try {
      payload = data.length > 0 ? asRecord(JSON.parse(data)) : {}
    } catch {
      // A non-JSON data frame is a protocol violation, but the terminal
      // event is what decides success; skip rather than kill the stream.
      continue
    }
    switch (event) {
      case 'response.output_item.added': {
        const item = asRecord(payload.item)
        const id = typeof item.id === 'string' ? item.id : ''
        if (id.length === 0) break
        const kind: ItemState['kind'] | undefined = item.type === 'message'
          ? 'text'
          : item.type === 'reasoning'
            ? 'reasoning'
            : item.type === 'function_call'
              ? 'tool-call'
              : undefined
        if (kind === undefined) break
        const state: ItemState = {
          index: nextIndex++,
          kind,
          text: '',
          arguments: '',
          callId: CallId(typeof item.call_id === 'string' ? item.call_id : id),
        }
        items.set(id, state)
        yield { type: 'block-start', index: state.index, blockType: kind === 'text' ? 'text' : kind === 'reasoning' ? 'reasoning' : 'tool-call' }
        break
      }
      case 'response.output_text.delta': {
        const state = itemOf(String(payload.item_id ?? ''), 'text')
        if (state === undefined) break
        const delta = String(payload.delta ?? '')
        state.text += delta
        if (delta.length > 0) yield { type: 'text-delta', index: state.index, text: delta }
        break
      }
      case 'response.reasoning_summary_text.delta':
      case 'response.reasoning_text.delta': {
        const state = itemOf(String(payload.item_id ?? ''), 'reasoning')
        if (state === undefined) break
        const delta = String(payload.delta ?? '')
        state.text += delta
        if (delta.length > 0) yield { type: 'reasoning-delta', index: state.index, text: delta }
        break
      }
      case 'response.function_call_arguments.delta': {
        const state = itemOf(String(payload.item_id ?? ''), 'tool-call')
        if (state === undefined) break
        const delta = String(payload.delta ?? '')
        state.arguments += delta
        yield { type: 'tool-call-delta', index: state.index, id: state.callId, argumentsDelta: delta }
        break
      }
      case 'response.output_item.done': {
        const item = asRecord(payload.item)
        const id = typeof item.id === 'string' ? item.id : ''
        if (item.type === 'message') {
          const state = id.length > 0 ? items.get(id) : undefined
          const parts = Array.isArray(item.content) ? item.content : []
          const text = typeof (asRecord(parts[0]).text) === 'string'
            ? String(asRecord(parts[0]).text)
            : (state?.text ?? '')
          const index = state?.index ?? nextIndex++
          yield { type: 'block-end', index, block: { type: 'text', text } satisfies ContentBlock }
        } else if (item.type === 'reasoning') {
          const state = id.length > 0 ? items.get(id) : undefined
          const summary = Array.isArray(item.summary)
            ? item.summary.map(part => String(asRecord(part).text ?? '')).join('')
            : ''
          const text = (state?.text ?? '') || summary
          const index = state?.index ?? nextIndex++
          yield { type: 'block-end', index, block: { type: 'reasoning', text } satisfies ContentBlock }
        } else if (item.type === 'function_call') {
          const state = id.length > 0 ? items.get(id) : undefined
          const callId = typeof item.call_id === 'string' ? item.call_id : ''
          const name = typeof item.name === 'string' ? item.name : ''
          const args = typeof item.arguments === 'string' ? item.arguments : (state?.arguments ?? '')
          const index = state?.index ?? nextIndex++
          sawToolCall = true
          yield {
            type: 'block-end',
            index,
            block: { type: 'tool-call', id: CallId(callId), name, arguments: args } satisfies ContentBlock,
          }
        }
        break
      }
      case 'response.incomplete': {
        const response = asRecord(payload.response)
        const usage = tokenUsageOf(asRecord(response.usage) as WireUsage)
        yield { type: 'usage', usage }
        const reason = String(asRecord(response.incomplete_details).reason ?? '')
        yield {
          type: 'finish',
          reason: reason === 'max_output_tokens' ? { kind: 'max-tokens' } : { kind: 'stop' },
        }
        return
      }
      case 'response.completed': {
        const response = asRecord(payload.response)
        yield { type: 'usage', usage: tokenUsageOf(asRecord(response.usage) as WireUsage) }
        yield { type: 'finish', reason: sawToolCall ? { kind: 'tool-calls' } : { kind: 'stop' } }
        return
      }
      case 'response.failed': {
        const error = asRecord(asRecord(payload.response).error)
        throw new LlmError(
          String(error.message ?? 'Responses stream failed'),
          wireErrorCode(typeof error.code === 'string' ? error.code : undefined, typeof error.message === 'string' ? error.message : undefined),
        )
      }
      case 'error': {
        const code = typeof payload.code === 'string' ? payload.code : undefined
        const message = typeof payload.message === 'string' ? payload.message : 'Responses stream error event'
        throw new LlmError(message, wireErrorCode(code, message))
      }
      default:
        // lifecycle bookkeeping events (created/in_progress/...) carry no
        // chunk vocabulary; ignore.
        break
    }
  }
  throw new LlmError('Responses SSE stream ended without a terminal event', 'STREAM_CLOSED')
}
