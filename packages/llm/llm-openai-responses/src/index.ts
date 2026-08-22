/**
 * Register a pure OpenAI **Responses API** adapter for the `openai-responses`
 * provider route on `ctx.llm`. This is the responses-side twin of llm-openai:
 * plain `POST {baseURL}/responses` with only fields every Responses endpoint
 * accepts. Nothing from the newest OpenAI surface (`prompt_cache_retention`,
 * `prompt_cache_options`, `service_tier`, `include`) is ever put on the wire —
 * gateways in front of volc/minimax/kimi reject those with a bare "Invalid
 * request Error", which is the failure that motivated this adapter.
 *
 * Configuration is entry-owned (no settings hot-swap section): pin baseURL,
 * models, and thinking policy through a profile patch (ds2 → one-api
 * coding_resps). Reasoning effort maps onto `reasoning.effort` with
 * `summary: "auto"`; `thinking: disabled` strips reasoning from the wire.
 *
 * @module @deepseek-ai/dsh-llm-openai-responses
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { assertUsableApiKey, LlmError, resolveRetryPolicy, RetryPolicySchema } from '@deepseek-ai/dsh-llm'
import type { RetryPolicyConfig } from '@deepseek-ai/dsh-llm'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import { launchEnvironmentOf, type LaunchEnvironmentSnapshot } from '@deepseek-ai/dsh-launch-environment'
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import { getOrCreateAnonymousUserId, type AnonymousUserId } from '@deepseek-ai/dsh-anonymous-user-id'
import { OpenAiResponsesAdapter, type ResponsesCatalogModel, type ResponsesConnectionOptions } from './adapter.ts'

export { OpenAiResponsesAdapter } from './adapter.ts'
export type { ResponsesCatalogModel, ResponsesConnectionOptions } from './adapter.ts'
export { buildResponsesRequest, wireEffortOf } from './wire.ts'
export { parseResponsesSse, translateResponses } from './translate.ts'
export type { ResponsesSseEvent } from './translate.ts'

export const name = 'llm-openai-responses'
export const inject = ['llm']

/** The single provider route this plugin owns. */
const PROVIDER = 'openai-responses'

/** Default maximum idle interval while an adapter stream read is outstanding. */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000
/** Default combined request/response context capacity. */
export const DEFAULT_CONTEXT_WINDOW = 262_144
/** Default per-request output-token cap. */
export const DEFAULT_MAX_TOKENS = 64_000
const DEFAULT_API_KEY_ENV = 'OPENAI_API_KEY'
const BASE_URL_ENV = 'OPENAI_BASE_URL'
export const PUBLIC_BASE_URL = 'https://api.openai.com/v1'

export interface Config {
  apiKeyEnv?: string
  baseURL?: string
  thinking?: 'enabled' | 'disabled'
  maxTokens?: number
  defaultContextWindow?: number
  models?: ResponsesCatalogModel[]
  streamIdleTimeoutMs?: number
  retryPolicy?: RetryPolicyConfig
}

const catalogModel: z<ResponsesCatalogModel> = z.object({
  id: z.string().required(),
  name: z.string(),
  description: z.string(),
  contextWindow: z.number().step(1).min(1),
  maxTokens: z.number().step(1).min(1),
})

export const Config: z<Config> = z.object({
  apiKeyEnv: z.string().role('credential-ref').default(DEFAULT_API_KEY_ENV),
  baseURL: z.string(),
  thinking: z.union(['enabled', 'disabled']),
  maxTokens: z.number().step(1).min(1).max(Number.MAX_SAFE_INTEGER).default(DEFAULT_MAX_TOKENS),
  defaultContextWindow: z.number().step(1).min(1).default(DEFAULT_CONTEXT_WINDOW),
  models: z.array(catalogModel).default([]),
  streamIdleTimeoutMs: z.number().min(Number.MIN_VALUE).max(MAX_TIMER_DELAY_MS).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
  retryPolicy: RetryPolicySchema,
})

/**
 * The one explicit resolve step from raw config to validated connection
 * facts; fails loud on programmatic misconstruction.
 */
export function resolveAdapterOptions(config: Config, environment?: LaunchEnvironmentSnapshot): ResponsesConnectionOptions {
  const maxTokens = config.maxTokens ?? DEFAULT_MAX_TOKENS
  if (!Number.isSafeInteger(maxTokens) || maxTokens <= 0) {
    throw new Error('llm-openai-responses: maxTokens must be a positive safe integer')
  }
  const defaultContextWindow = config.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW
  if (!Number.isInteger(defaultContextWindow) || defaultContextWindow <= 0) {
    throw new Error('llm-openai-responses: defaultContextWindow must be a positive integer')
  }
  const streamIdleTimeoutMs = config.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS
  if (!Number.isFinite(streamIdleTimeoutMs) || streamIdleTimeoutMs <= 0 || streamIdleTimeoutMs > MAX_TIMER_DELAY_MS) {
    throw new Error(
      `llm-openai-responses: streamIdleTimeoutMs must be a positive finite number no greater than ${MAX_TIMER_DELAY_MS}`,
    )
  }
  const models = (config.models ?? []).map((model) => {
    if (model.id.length === 0) throw new Error('llm-openai-responses: catalog model ids must be non-empty')
    if (model.contextWindow !== undefined && (!Number.isInteger(model.contextWindow) || model.contextWindow <= 0)) {
      throw new Error(`llm-openai-responses: catalog model "${model.id}" contextWindow must be a positive integer`)
    }
    return model
  })
  return {
    apiKeyEnv: credentialRef(config.apiKeyEnv ?? DEFAULT_API_KEY_ENV),
    baseURL: config.baseURL
      ?? environment?.get(BASE_URL_ENV)?.value
      ?? PUBLIC_BASE_URL,
    thinking: config.thinking,
    maxTokens,
    defaultContextWindow,
    models,
    streamIdleTimeoutMs,
    retryPolicy: resolveRetryPolicy(config.retryPolicy, 'llm-openai-responses: retryPolicy'),
  }
}

export function apply(ctx: Context, config: Config): void {
  const options = (): ResponsesConnectionOptions => resolveAdapterOptions(config, launchEnvironmentOf(ctx))
  options()

  const resolveApiKey = async (connection: ResponsesConnectionOptions): Promise<string> => {
    const ref = connection.apiKeyEnv
    const credentials = ctx.get('credentials')
    if (credentials !== undefined) {
      const hit = await credentials.resolve(ref)
      if (hit !== undefined) return assertUsableApiKey(hit.value, 'llm-openai-responses', ref)
    } else {
      const ambient = launchEnvironmentOf(ctx).get(ref)
      if (ambient !== undefined && ambient.value.length > 0) {
        return assertUsableApiKey(ambient.value, 'llm-openai-responses', ref)
      }
    }
    throw new LlmError(
      `llm-openai-responses: no API key for provider route "${PROVIDER}"; export ${ref} in the launching environment`,
      'MISSING_CREDENTIAL',
    )
  }

  let userId: AnonymousUserId | undefined
  const adapter = new OpenAiResponsesAdapter({
    options,
    resolveApiKey,
    resolveUserId: () => userId ??= getOrCreateAnonymousUserId(),
  })
  ctx.llm.registerAdapter([PROVIDER], adapter)
}
