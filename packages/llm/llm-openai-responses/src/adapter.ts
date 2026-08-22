/**
 * `OpenAiResponsesAdapter`: fetch + SSE against a pure OpenAI Responses API
 * endpoint (`POST {baseURL}/responses`), emitting harness StreamChunks.
 * Transport-only, mirroring the DeepSeek adapter's shape: connection facts
 * arrive through a per-operation thunk, the bearer token through a
 * per-request resolver, and one stable abort signal reaches both the initial
 * fetch and every body read behind a per-read idle watchdog.
 *
 * @module dsh-llm-openai-responses/adapter
 */

import {
  attributionHeaders,
  LlmAdapter,
  LlmError,
  ProviderRequestId,
  ReasoningEffortId,
} from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  ResolvedRetryPolicy,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import type { CredentialRef } from '@deepseek-ai/dsh-credentials'
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout'
import type { AnonymousUserId } from '@deepseek-ai/dsh-anonymous-user-id'
import { buildResponsesRequest } from './wire.ts'
import { parseResponsesSse, translateResponses } from './translate.ts'

/** One optional model entry advertised by this adapter (chat-route catalog shape). */
export interface ResponsesCatalogModel {
  id: string
  name?: string
  description?: string
  contextWindow?: number
  maxTokens?: number
}

/**
 * Validated connection facts for one operation; the registering plugin's
 * resolve step produces this shape once per call.
 */
export interface ResponsesConnectionOptions {
  baseURL: string
  apiKeyEnv: CredentialRef
  thinking: 'enabled' | 'disabled' | undefined
  maxTokens: number
  defaultContextWindow: number
  models: readonly ResponsesCatalogModel[]
  streamIdleTimeoutMs: number
  retryPolicy: ResolvedRetryPolicy
}

/** Constructor hooks the registering plugin owns. */
export interface OpenAiResponsesAdapterOptions {
  options: () => ResponsesConnectionOptions
  resolveApiKey: (connection: ResponsesConnectionOptions) => Promise<string>
  resolveUserId: () => AnonymousUserId
}

const STREAM_IDLE_TIMEOUT_CODE = 'LLM_STREAM_IDLE_TIMEOUT'
const OFF_REASONING_EFFORT = ReasoningEffortId('off')
const LOW_REASONING_EFFORT = ReasoningEffortId('low')
const MEDIUM_REASONING_EFFORT = ReasoningEffortId('medium')
const HIGH_REASONING_EFFORT = ReasoningEffortId('high')
const MAX_REASONING_EFFORT = ReasoningEffortId('max')
const REASONING_EFFORTS = [
  { id: OFF_REASONING_EFFORT, name: 'Off' },
  { id: LOW_REASONING_EFFORT, name: 'Low' },
  { id: MEDIUM_REASONING_EFFORT, name: 'Medium' },
  { id: HIGH_REASONING_EFFORT, name: 'High' },
  { id: MAX_REASONING_EFFORT, name: 'Max' },
] as const
const OFF_ONLY_REASONING_EFFORTS = [
  { id: OFF_REASONING_EFFORT, name: 'Off' },
] as const

function modelInfo(provider: string, model: ResponsesCatalogModel): LlmModelInfo {
  return {
    provider,
    id: model.id,
    name: model.name ?? model.id,
    ...model.description === undefined ? {} : { description: model.description },
    inputModalities: ['text'],
  }
}

function providerRetryAfterMs(value: string | null): number | undefined {
  if (value === null) return undefined
  if (/^\d+$/.test(value)) {
    const delay = Number(value) * 1_000
    return Number.isFinite(delay) && delay > 0 ? delay : undefined
  }
  const delay = Date.parse(value) - Date.now()
  return Number.isFinite(delay) && delay > 0 ? delay : undefined
}

/**
 * Map an HTTP status to a stable LlmError code for the Responses wire.
 * @param status - status of a non-2xx provider response.
 * @param code - parsed provider error code, when available.
 * @param message - parsed provider error message, when available.
 */
export function responsesHttpErrorCode(status: number, code?: string, message?: string): string {
  if (status === 401 || status === 403) return 'AUTH'
  if (status === 429) return 'RATE_LIMIT'
  if (status === 400) {
    const detail = [code, message].filter(Boolean).join(' ')
    if (detail.toLowerCase().includes('context length') || detail.toLowerCase().includes('context window')) {
      return 'CONTEXT_WINDOW_EXCEEDED'
    }
    return 'INVALID_REQUEST'
  }
  if (status >= 500) return 'SERVER'
  return `HTTP_${status}`
}

/**
 * Pure OpenAI Responses adapter. One instance serves every model name it was
 * registered under; the harness model name IS the wire model name.
 */
export class OpenAiResponsesAdapter extends LlmAdapter {
  constructor(private readonly config: OpenAiResponsesAdapterOptions) {
    super()
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'OpenAI Responses' }
  }

  override providerRetryPolicy(_provider: string): ResolvedRetryPolicy {
    return this.config.options().retryPolicy
  }

  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    return Promise.resolve(this.config.options().models.map(model => modelInfo(provider, model)))
  }

  override resolveModel(
    provider: string,
    model: string,
    _signal?: AbortSignal,
  ): Promise<LlmResolvedModelInfo> {
    const connection = this.config.options()
    const configured = connection.models.find(entry => entry.id === model)
    const contextWindow = configured?.contextWindow ?? connection.defaultContextWindow
    return Promise.resolve({
      ...configured === undefined
        ? { provider, id: model, name: model, inputModalities: ['text' as const] }
        : modelInfo(provider, configured),
      context: { contextWindow },
      defaultMaxTokens: configured?.maxTokens ?? connection.maxTokens,
      ...connection.thinking === 'disabled'
        ? {
          reasoning: {
            efforts: OFF_ONLY_REASONING_EFFORTS,
            defaultEffort: OFF_REASONING_EFFORT,
          },
        }
        : {
          reasoning: {
            efforts: REASONING_EFFORTS,
            defaultEffort: HIGH_REASONING_EFFORT,
          },
        },
    })
  }

  async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const connection = this.config.options()
    const apiKey = await this.config.resolveApiKey(connection)
    const userId = this.config.resolveUserId()
    const consumer = new AbortController()
    const upstream = options.signal === undefined
      ? consumer.signal
      : AbortSignal.any([options.signal, consumer.signal])
    using watchdog = idleWatchdog(upstream, connection.streamIdleTimeoutMs, STREAM_IDLE_TIMEOUT_CODE)
    const iterator = this.request(options, watchdog.signal, connection, apiKey, userId, () => { watchdog.pulse() })[Symbol.asyncIterator]()
    let exhausted = false
    try {
      while (true) {
        const result = await watchdog.next(iterator)
        if (result.done) {
          exhausted = true
          return
        }
        yield result.value
      }
    } catch (error: unknown) {
      if (timeoutOf(watchdog.signal, STREAM_IDLE_TIMEOUT_CODE) !== undefined) {
        throw new LlmError(
          `Responses stream idle timeout after ${connection.streamIdleTimeoutMs}ms`,
          'TIMEOUT',
          { cause: error },
        )
      }
      if (options.signal?.aborted) {
        throw new LlmError('Responses request aborted by caller', 'ABORTED', { cause: error })
      }
      if (error instanceof LlmError) throw error
      throw new LlmError(`Responses API stream from ${connection.baseURL} failed`, 'TRANSPORT', { cause: error })
    } finally {
      consumer.abort('Responses stream consumer stopped')
      if (!exhausted && iterator.return !== undefined) {
        try {
          await iterator.return()
        } catch {
          // The consumer controller already owns termination.
        }
      }
    }
  }

  private async * request(
    options: GenerateOptions,
    signal: AbortSignal,
    connection: ResponsesConnectionOptions,
    apiKey: string,
    userId: AnonymousUserId,
    onComment: () => void,
  ): AsyncIterable<StreamChunk> {
    const body = buildResponsesRequest(options, connection.thinking)
    const payload = JSON.stringify(body)
    const headers = {
      'authorization': `Bearer ${apiKey}`,
      'content-type': 'application/json',
      'accept': 'text/event-stream',
      ...attributionHeaders(),
      'x-deepseek-harness-user-id': String(userId),
      ...options.sessionId !== undefined
        ? { 'x-deepseek-harness-session-id': String(options.sessionId) }
        : {},
      ...options.purpose === 'compaction'
        ? { 'x-deepseek-harness-compact': '1' }
        : {},
    }

    let response: Response
    try {
      response = await fetch(`${connection.baseURL}/responses`, {
        method: 'POST',
        headers,
        body: payload,
        signal,
      })
    } catch (error: unknown) {
      if (signal.aborted) throw error
      throw new LlmError(
        `Responses API request to ${connection.baseURL} failed`,
        'TRANSPORT',
        { cause: error },
      )
    }

    if (!response.ok) {
      let message = `Responses API error (HTTP ${response.status})`
      let code: string | undefined
      try {
        const parsed = await response.json() as { error?: { code?: string; message?: string } }
        if (parsed.error?.message !== undefined) message = parsed.error.message
        code = parsed.error?.code
      } catch {
        // Malformed gateway JSON must not mask the HTTP status.
      }
      const delay = providerRetryAfterMs(response.headers.get('retry-after'))
      const id = response.headers.get('x-request-id')
      throw new LlmError(message, responsesHttpErrorCode(response.status, code, message), {
        status: response.status,
        ...delay === undefined ? {} : { providerRetryAfterMs: delay },
        ...id === null || id.length === 0 ? {} : { requestId: ProviderRequestId(id) },
      })
    }
    if (!response.body) {
      throw new LlmError('Responses API returned no response body', 'EMPTY_RESPONSE')
    }

    yield* translateResponses(parseResponsesSse(response.body, onComment))
  }
}
