/**
 * Real-composition guard for the llm-openai route: the plugin boots through
 * the actual Loader + Include path next to LlmRuntime (and optionally
 * dsh-credentials-local), streams text and tool calls off a mock
 * chat-completions endpoint, and — the point of the route — replays an
 * assembled tool-call history whose `function.arguments` travels on the wire
 * as a JSON OBJECT with gateway extras riding under their own keys and no
 * DeepSeek-specific fields anywhere.
 */

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import Loader from '@deepseek-ai/cordis-plugin-loader'
import Include from '@deepseek-ai/cordis-plugin-include'
import LlmRuntime from '@deepseek-ai/dsh-llm'
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
import type { FinishReason, GenerateOptions, Message } from '@deepseek-ai/dsh-llm'
import LocalCredentialProvider from '@deepseek-ai/dsh-credentials-local'
import * as LlmOpenAi from '../src/index.ts'
import { closeMockServers, mockServer, textEvents } from '../../llm-deepseek/tests/mock-server.ts'

const REQUEST_EXTRAS = { chat_template_kwargs: { enable_thinking: false } }
/** Tool-call stream: one call assembled from argument fragments. */
const toolCallEvents = [
  '{"choices":[{"delta":{"role":"assistant","content":null}}]}',
  '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}}]}}]}',
  '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"command\\":\\"echo openai" }}]}}]}',
  '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"-style\\"}"}}]}}]}',
  '{"choices":[{"delta":{"content":""},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":7}}',
  '[DONE]',
]

let root: string | undefined
let context: Context | undefined

afterEach(async () => {
  await context?.fiber.dispose()
  context = undefined
  if (root !== undefined) await rm(root, { recursive: true, force: true })
  root = undefined
  await closeMockServers()
  vi.unstubAllEnvs()
})

interface Composition {
  ctx: Context
}

async function loadComposition(options: {
  baseURL: string
  withCredentials: boolean | 'empty'
  requestExtras?: Record<string, unknown>
}): Promise<Composition> {
  root = await mkdtemp(join(tmpdir(), 'dsh-llm-openai-'))
  vi.stubEnv('DSH_HOME', root)
  if (options.withCredentials === true) {
    await writeFile(join(root, '.credentials.yaml'), 'OPENAI_API_KEY: file-key\n', { mode: 0o600 })
  }
  const configPath = join(root, 'cordis.yml')
  await writeFile(configPath, [
    '- id: llm',
    "  name: 'test-llm-service'",
    ...options.withCredentials !== false
      ? [
        '- id: credentials',
        "  name: '@deepseek-ai/dsh-credentials-local'",
        '  config:',
        `    path: ${JSON.stringify(join(root, '.credentials.yaml'))}`,
        '    debounceMs: 10',
      ]
      : [],
    '- id: llm-openai',
    "  name: '@deepseek-ai/dsh-llm-openai'",
    '  config:',
    `    baseURL: ${JSON.stringify(options.baseURL)}`,
    '    thinking: disabled',
    ...options.requestExtras === undefined ? [] : [`    requestExtras: ${JSON.stringify(options.requestExtras)}`],
    '',
  ].join('\n'))

  const ctx = new Context()
  context = ctx
  ctx.baseUrl = pathToFileURL(root).href + '/'
  await ctx.plugin(Loader)
  ctx.loader.builtins.include = Include
  const modules = new Map<string, unknown>([
    ['test-llm-service', LlmRuntime],
    ['@deepseek-ai/dsh-credentials-local', LocalCredentialProvider],
    ['@deepseek-ai/dsh-llm-openai', LlmOpenAi],
  ])
  ctx.loader.internal = {
    version: 'v2',
    async import(specifier: string) {
      if (!modules.has(specifier)) throw new Error(`unexpected Loader import: ${specifier}`)
      return modules.get(specifier)
    },
  } as unknown as NonNullable<typeof ctx.loader.internal>
  await ctx.loader.create({
    name: 'cordis:include',
    config: { path: pathToFileURL(configPath).href },
  })
  await ctx.loader.await()
  return { ctx }
}

/** The slice of a captured request body the wire assertions read. */
interface CapturedRequest {
  messages: { role: string; tool_calls?: { function: { arguments: unknown } }[] }[]
  chat_template_kwargs?: unknown
  thinking?: unknown
  reasoning_effort?: unknown
}

async function assemble(ctx: Context, options: Omit<GenerateOptions, 'provider'>): Promise<{ message: Message; finish: FinishReason }> {
  const assembler = new BlockAssembler()
  const request = { provider: 'openai', ...options }
  for await (const chunk of ctx.llm.stream(request)) assembler.push(chunk)
  return {
    message: assembler.message({ kind: 'model', provider: 'openai', model: request.model }),
    finish: assembler.finish,
  }
}

describe('llm-openai real composition', () => {
  it('streams tool calls, then replays their history with OBJECT-style arguments and gateway extras on the wire', async () => {
    vi.stubEnv('OPENAI_API_KEY', 'entry-key')
    const server = await mockServer([
      { kind: 'sse', events: toolCallEvents },
      { kind: 'sse', events: textEvents },
      { kind: 'sse', events: textEvents },
    ])
    const { ctx } = await loadComposition({
      baseURL: server.url,
      withCredentials: false,
      requestExtras: REQUEST_EXTRAS,
    })

    const { message } = await assemble(ctx, { model: 'qwen3.8-27b', messages: [] })
    expect(message.content).toEqual([
      { type: 'tool-call', id: 'call_1', name: 'bash', arguments: '{"command":"echo openai-style"}' },
    ])
    expect(server.headers[0]?.authorization).toBe('Bearer entry-key')

    // Replay the assembled tool-call turn as history — the pure-OpenAI wire
    // assertion: arguments is a JSON object, extras ride along, no
    // DeepSeek-specific fields anywhere.
    await assemble(ctx, { model: 'qwen3.8-27b', messages: [message] })
    const replay = server.requests[1] as CapturedRequest
    expect(replay.messages[0]?.role).toBe('assistant')
    expect(replay.messages[0]?.tool_calls?.[0]?.function.arguments).toEqual({ command: 'echo openai-style' })
    expect(replay.chat_template_kwargs).toEqual({ enable_thinking: false })
    expect('thinking' in replay).toBe(false)
    expect('reasoning_effort' in replay).toBe(false)
    expect('reasoning_content' in replay.messages[0]!).toBe(false)

    // A second request on the same boot also exercises the memoized user id.
    await assemble(ctx, { model: 'qwen3.8-27b', messages: [] })
    expect(server.requests).toHaveLength(3)
  })

  it('resolves the credential from the credentials service when the seam is present', async () => {
    vi.stubEnv('OPENAI_API_KEY', '')
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const { ctx } = await loadComposition({ baseURL: server.url, withCredentials: true })
    await assemble(ctx, { model: 'qwen3.8-27b', messages: [] })
    expect(server.requests).toHaveLength(1)
    expect(server.headers[0]?.authorization).toBe('Bearer file-key')
  })

  it('fails with MISSING_CREDENTIAL when no seam and no environment key exists', async () => {
    vi.stubEnv('OPENAI_API_KEY', '')
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const { ctx } = await loadComposition({ baseURL: server.url, withCredentials: false })
    // Adapter dispatch failures surface as the terminal error-finish chunk,
    // never as a thrown stream error (the service boundary's contract).
    // Both absence shapes count: the empty-string ambient value…
    const empty = await assemble(ctx, { model: 'qwen3.8-27b', messages: [] })
    expect(empty.finish).toMatchObject({ kind: 'error', failure: { code: 'MISSING_CREDENTIAL' } })
    // …and a variable that is not set at all.
    delete process.env.OPENAI_API_KEY
    const absent = await assemble(ctx, { model: 'qwen3.8-27b', messages: [] })
    expect(absent.finish).toMatchObject({ kind: 'error', failure: { code: 'MISSING_CREDENTIAL' } })
    expect(server.requests).toHaveLength(0)
  })

  it('fails with MISSING_CREDENTIAL when the credentials seam holds no key', async () => {
    vi.stubEnv('OPENAI_API_KEY', '')
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    // The seam is present but its document stores nothing: resolution must
    // fall through to the terminal error rather than send a keyless request.
    const { ctx } = await loadComposition({ baseURL: server.url, withCredentials: 'empty' })
    const { finish } = await assemble(ctx, { model: 'qwen3.8-27b', messages: [] })
    expect(finish).toMatchObject({ kind: 'error', failure: { code: 'MISSING_CREDENTIAL' } })
    expect(server.requests).toHaveLength(0)
  })
})
