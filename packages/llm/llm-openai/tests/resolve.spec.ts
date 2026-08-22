/**
 * Unit guard for entry-config resolution: every beyond-schema bound fails
 * loud with its own message, defaults apply per field, the endpoint falls
 * back config → trusted environment → public API, and the advisory catalog
 * validates ids/names/caps/duplicates.
 */

import { describe, expect, it } from 'vitest'
import type { LaunchEnvironmentSnapshot } from '@deepseek-ai/dsh-launch-environment'
import {
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_TOKENS,
  DEFAULT_STREAM_IDLE_TIMEOUT_MS,
  PUBLIC_BASE_URL,
  resolveAdapterOptions,
} from '../src/index.ts'

const environmentWith = (entries: Record<string, string>): LaunchEnvironmentSnapshot =>
  ({ get: (name: string) => entries[name] === undefined ? undefined : { value: entries[name] } }) as unknown as LaunchEnvironmentSnapshot

describe('resolveAdapterOptions bounds', () => {
  it('applies documented defaults to a minimal config', () => {
    const resolved = resolveAdapterOptions({})
    expect(resolved.baseURL).toBe(PUBLIC_BASE_URL)
    expect(resolved.maxTokens).toBe(DEFAULT_MAX_TOKENS)
    expect(resolved.defaultContextWindow).toBe(DEFAULT_CONTEXT_WINDOW)
    expect(resolved.streamIdleTimeoutMs).toBe(DEFAULT_STREAM_IDLE_TIMEOUT_MS)
    expect(resolved.models).toEqual([])
    expect(resolved.defaults).toEqual({ thinking: undefined, reasoningEffort: undefined })
    expect(resolved.apiKeyEnv).toBe('OPENAI_API_KEY')
  })

  it('keeps configured values and maps thinking to the advertised-effort flag only', () => {
    const resolved = resolveAdapterOptions({
      baseURL: 'http://127.0.0.1:10001/v1',
      apiKeyEnv: 'LOCAL_KEY',
      thinking: 'disabled',
      maxTokens: 32768,
      defaultContextWindow: 131072,
      streamIdleTimeoutMs: 60_000,
    })
    expect(resolved.baseURL).toBe('http://127.0.0.1:10001/v1')
    expect(resolved.apiKeyEnv).toBe('LOCAL_KEY')
    expect(resolved.defaults.thinking).toBe('disabled')
    expect(resolved.defaults.reasoningEffort).toBeUndefined()
    expect(resolved.maxTokens).toBe(32_768)
    expect(resolved.defaultContextWindow).toBe(131_072)
    expect(resolved.streamIdleTimeoutMs).toBe(60_000)
  })

  it('resolves the endpoint from a trusted environment layer when config is silent', () => {
    expect(resolveAdapterOptions({}, environmentWith({ OPENAI_BASE_URL: 'http://env:9/v1' })).baseURL)
      .toBe('http://env:9/v1')
  })

  it('rejects out-of-bound numeric fields', () => {
    expect(() => resolveAdapterOptions({ maxTokens: 0 })).toThrow('llm-openai: maxTokens must be a positive safe integer')
    expect(() => resolveAdapterOptions({ defaultContextWindow: 0 })).toThrow('llm-openai: defaultContextWindow must be a positive integer')
    expect(() => resolveAdapterOptions({ streamIdleTimeoutMs: 0 })).toThrow(/streamIdleTimeoutMs must be a positive finite number/)
  })

  it('rejects a non-plain-object requestExtras', () => {
    expect(() => resolveAdapterOptions({ requestExtras: ['nope'] as unknown as Record<string, unknown> }))
      .toThrow('llm-openai: requestExtras must be a plain object of extra request body fields')
  })
})

describe('catalog validation', () => {
  it('accepts a well-formed catalog and keeps it verbatim', () => {
    const models = [{ id: 'qwen3.8-27b', name: 'Qwen3.8 27B', contextWindow: 131_072, maxTokens: 32_768 }]
    expect(resolveAdapterOptions({ models }).models).toEqual(models)
  })

  it('rejects empty ids and names', () => {
    expect(() => resolveAdapterOptions({ models: [{ id: '' }] })).toThrow('llm-openai: catalog model ids must be non-empty')
    expect(() => resolveAdapterOptions({ models: [{ id: 'm', name: '' }] })).toThrow('catalog model "m" has an empty name')
  })

  it('rejects non-positive caps and duplicate ids', () => {
    expect(() => resolveAdapterOptions({ models: [{ id: 'm', contextWindow: 0 }] }))
      .toThrow('catalog model "m" contextWindow must be a positive integer')
    expect(() => resolveAdapterOptions({ models: [{ id: 'm', maxTokens: 0 }] }))
      .toThrow('catalog model "m" maxTokens must be a positive integer')
    expect(() => resolveAdapterOptions({ models: [{ id: 'm' }, { id: 'm' }] })).toThrow('duplicate catalog model "m"')
  })
})
