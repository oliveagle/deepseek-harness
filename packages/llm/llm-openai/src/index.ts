/**
 * Register a pure OpenAI-style chat-completions adapter for the `openai`
 * provider route on `ctx.llm`. The adapter reuses the DeepSeek transport
 * (fetch + SSE + idle watchdog + error mapping, via `DeepSeekAdapter` with an
 * injected serializer) and differs from the native DeepSeek route in exactly
 * two wire behaviors:
 *
 * 1. assistant history `tool_calls[].function.arguments` travels as a JSON
 *    OBJECT (see serialize.ts) — gateways that type the field as an object
 *    accept the request without re-parsing, and malformed arguments degrade
 *    to `{}` instead of bricking later turns;
 * 2. no DeepSeek-specific fields ever reach the wire (`reasoning_content`,
 *    `thinking`, `reasoning_effort` are dropped), so the route speaks plain
 *    OpenAI chat completions to any compatible endpoint.
 *
 * Configuration is entry-owned (no settings hot-swap section): connection
 * facts re-resolve from the entry config on every operation, which is the
 * documented shape for a patch-pinned direct route like `ds local` → local
 * sglang. `requestExtras` carries gateway-specific body fields verbatim —
 * e.g. sglang's `chat_template_kwargs: {enable_thinking: false}` — merged
 * under core request fields.
 * @module @deepseek-ai/dsh-llm-openai
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { assertUsableApiKey, LlmError, resolveRetryPolicy, RetryPolicySchema } from '@deepseek-ai/dsh-llm'
import type { RetryPolicyConfig } from '@deepseek-ai/dsh-llm'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import { launchEnvironmentOf, type LaunchEnvironmentSnapshot } from '@deepseek-ai/dsh-launch-environment'
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import { getOrCreateAnonymousUserId, type AnonymousUserId } from '@deepseek-ai/dsh-anonymous-user-id'
import { DeepSeekAdapter, type DeepSeekCatalogModel, type DeepSeekConnectionOptions } from '@deepseek-ai/dsh-llm-deepseek'
import { serializeRequest } from './serialize.ts'

export { objectArguments, serializeRequest, toObjectStyle } from './serialize.ts'
export type * from './types.ts'

export const name = 'llm-openai'
export const inject = ['llm']

/** The single provider route this plugin owns. */
const PROVIDER = 'openai'

/** Default maximum idle interval while an adapter stream read is outstanding. */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000
/** Default combined request/response context capacity. */
export const DEFAULT_CONTEXT_WINDOW = 131_072
/** Default per-request output-token cap. */
export const DEFAULT_MAX_TOKENS = 32_768
const DEFAULT_API_KEY_ENV = 'OPENAI_API_KEY'
/** Environment variable naming this provider's endpoint, honored only from trusted layers. */
const BASE_URL_ENV = 'OPENAI_BASE_URL'
/** Endpoint when neither config nor environment names one. */
export const PUBLIC_BASE_URL = 'https://api.openai.com/v1'

/** One optional model entry advertised by the adapter (same catalog shape as the DeepSeek route). */
export type OpenAiCatalogModel = DeepSeekCatalogModel

/**
 * Plugin config. Every field is optional in yml: a missing API key resolves
 * through {@link Config.apiKeyEnv} at each request (a request without any key
 * fails with `MISSING_CREDENTIAL`, not at plugin load).
 */
export interface Config {
  /** Credential reference (environment-variable name) resolved per request; defaults to `OPENAI_API_KEY`. */
  apiKeyEnv?: string
  /** Endpoint base (`/chat/completions` is appended); falls back to $OPENAI_BASE_URL from a
   * trusted environment layer, then the public API. */
  baseURL?: string
  /** Thinking policy. `disabled` advertises effort Off only; nothing thinking-related is ever sent on this wire. */
  thinking?: 'enabled' | 'disabled'
  /** Default per-request output cap; a model's own cap and explicit request values win. */
  maxTokens?: number
  /** Positive context capacity used when the selected model has no exact value. */
  defaultContextWindow?: number
  /** Advisory models shown by discovery consumers; defaults to none (the route accepts any model id). */
  models?: OpenAiCatalogModel[]
  /** Maximum provider idle time while one stream read is outstanding. */
  streamIdleTimeoutMs?: number
  /** Provider-owned model-request retry policy; omission uses normal defaults. */
  retryPolicy?: RetryPolicyConfig
  /**
   * Gateway-specific request body fields merged verbatim into every request
   * (e.g. sglang `chat_template_kwargs`); core request fields win same-key
   * collisions. Must be a plain object when present.
   */
  requestExtras?: Record<string, unknown>
}

const catalogModel: z<OpenAiCatalogModel> = z.object({
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
  requestExtras: z.dict(z.any()),
})

/** Validated connection facts for one operation (the DeepSeek transport's resolution shape). */
export type ResolvedOpenAiOptions = DeepSeekConnectionOptions

/**
 * The one explicit resolve step from raw config to validated connection
 * facts. Programmatic construction may bypass Schemastery normalization, so
 * every default and bound is re-judged here — and this plugin has no
 * settings hot-swap, so a failure always fails loud at its call site.
 */
export function resolveAdapterOptions(config: Config, environment?: LaunchEnvironmentSnapshot): ResolvedOpenAiOptions {
  const maxTokens = config.maxTokens ?? DEFAULT_MAX_TOKENS
  if (!Number.isSafeInteger(maxTokens) || maxTokens <= 0) {
    throw new Error('llm-openai: maxTokens must be a positive safe integer')
  }
  const defaultContextWindow = config.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW
  if (!Number.isInteger(defaultContextWindow) || defaultContextWindow <= 0) {
    throw new Error('llm-openai: defaultContextWindow must be a positive integer')
  }
  const streamIdleTimeoutMs = config.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS
  if (!Number.isFinite(streamIdleTimeoutMs) || streamIdleTimeoutMs <= 0 || streamIdleTimeoutMs > MAX_TIMER_DELAY_MS) {
    throw new Error(
      `llm-openai: streamIdleTimeoutMs must be a positive finite number no greater than ${MAX_TIMER_DELAY_MS}`,
    )
  }
  if (config.requestExtras !== undefined && !isPlainObject(config.requestExtras)) {
    throw new Error('llm-openai: requestExtras must be a plain object of extra request body fields')
  }
  return {
    apiKeyEnv: credentialRef(config.apiKeyEnv ?? DEFAULT_API_KEY_ENV),
    baseURL: config.baseURL
      ?? environment?.get(BASE_URL_ENV)?.value
      ?? PUBLIC_BASE_URL,
    defaults: {
      // Pure OpenAI has no thinking wire field; the flag only steers the
      // advertised effort set (Off-only when disabled) on the reused transport.
      thinking: config.thinking,
      reasoningEffort: undefined,
    },
    maxTokens,
    defaultContextWindow,
    models: resolveModels(config.models),
    streamIdleTimeoutMs,
    retryPolicy: resolveRetryPolicy(config.retryPolicy, 'llm-openai: retryPolicy'),
  }
}

/** Runtime plain-object guard: the schema already shapes yml, programmatic callers may not. */
function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Resolve, validate, and detach the advisory model catalog. */
function resolveModels(models: readonly OpenAiCatalogModel[] | undefined): OpenAiCatalogModel[] {
  const seen = new Set<string>()
  return (models ?? []).map((model) => {
    if (model.id.length === 0) throw new Error('llm-openai: catalog model ids must be non-empty')
    if (model.name !== undefined && model.name.length === 0) {
      throw new Error(`llm-openai: catalog model "${model.id}" has an empty name`)
    }
    if (model.contextWindow !== undefined
      && (!Number.isInteger(model.contextWindow) || model.contextWindow <= 0)) {
      throw new Error(`llm-openai: catalog model "${model.id}" contextWindow must be a positive integer`)
    }
    if (model.maxTokens !== undefined
      && (!Number.isInteger(model.maxTokens) || model.maxTokens <= 0)) {
      throw new Error(`llm-openai: catalog model "${model.id}" maxTokens must be a positive integer`)
    }
    if (seen.has(model.id)) throw new Error(`llm-openai: duplicate catalog model "${model.id}"`)
    seen.add(model.id)
    return model
  })
}

export function apply(ctx: Context, config: Config): void {
  // Fail loud at load: an invalid entry config must abort composition, not
  // surface as a per-request error later.
  const options = (): ResolvedOpenAiOptions => resolveAdapterOptions(config, launchEnvironmentOf(ctx))
  options()

  const resolveApiKey = async (connection: ResolvedOpenAiOptions): Promise<string> => {
    const ref = connection.apiKeyEnv
    const credentials = ctx.get('credentials')
    if (credentials !== undefined) {
      const hit = await credentials.resolve(ref)
      if (hit !== undefined) return assertUsableApiKey(hit.value, 'llm-openai', ref)
    } else {
      // Without the seam there is no managed store to rank against, so the
      // environment is the whole credential plane.
      const ambient = launchEnvironmentOf(ctx).get(ref)
      if (ambient !== undefined && ambient.value.length > 0) {
        return assertUsableApiKey(ambient.value, 'llm-openai', ref)
      }
    }
    throw new LlmError(
      `llm-openai: no API key for provider route "${PROVIDER}"; export ${ref} in the launching environment`,
      'MISSING_CREDENTIAL',
    )
  }

  let userId: AnonymousUserId | undefined
  const adapter = new DeepSeekAdapter({
    options,
    resolveApiKey,
    resolveUserId: () => userId ??= getOrCreateAnonymousUserId(),
    serialize: request => serializeRequest(request, config.requestExtras),
    providerName: 'OpenAI-compatible',
  })
  ctx.llm.registerAdapter([PROVIDER], adapter)
}
