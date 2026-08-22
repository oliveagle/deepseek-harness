/**
 * Unit guard for the pure-OpenAI serializer: object-style tool-call
 * arguments (plain parse, double-encoded unwrap, depth cap, junk fallback),
 * reasoning_content stripping, tool-result expansion, and the request shape
 * (system/tools/sampling optionality, requestExtras merge with core fields
 * winning collisions).
 */

import { describe, expect, it } from 'vitest'
import { CallId, createMessage } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, Message, ToolSchema } from '@deepseek-ai/dsh-llm'
import { objectArguments, serializeRequest, toObjectStyle } from '../src/serialize.ts'

const source = { kind: 'plugin' as const, plugin: 'test' }

const text = (role: Message['role'], content: string): Message =>
  createMessage({ role, content: [{ type: 'text', text: content }], source })

const assistantToolCall = (id: string, name: string, args: string): Message =>
  createMessage({
    role: 'assistant',
    content: [{ type: 'tool-call', id: CallId(id), name, arguments: args }],
    source,
  })

const toolResult = (callId: string, output: string): Message =>
  createMessage({
    role: 'user',
    content: [{ type: 'tool-result', toolCallId: CallId(callId), content: [{ type: 'text', text: output }] }],
    source,
  })

const tool: ToolSchema = {
  name: 'bash',
  description: 'Run a shell command',
  parameters: { type: 'object', properties: { command: { type: 'string' } }, required: ['command'] },
}

const baseOptions = (messages: Message[]): GenerateOptions => ({
  provider: 'openai',
  model: 'qwen3.8-27b',
  messages,
})

describe('objectArguments', () => {
  it('parses a plain JSON object string', () => {
    expect(objectArguments('{"command":"echo hi"}')).toEqual({ command: 'echo hi' })
  })

  it('unwraps a double-encoded quoted JSON string', () => {
    const inner = JSON.stringify({ objective: 'list the directory' })
    expect(objectArguments(JSON.stringify(inner))).toEqual({ objective: 'list the directory' })
  })

  it('falls back to {} on non-JSON text', () => {
    expect(objectArguments('file_path=/tmp/x content=y')).toEqual({})
    expect(objectArguments('')).toEqual({})
  })

  it('falls back to {} when wrapping exceeds the depth cap or lands on a non-object', () => {
    let deep = JSON.stringify({ k: 1 })
    for (let index = 0; index < 10; index += 1) deep = JSON.stringify(deep)
    expect(objectArguments(deep)).toEqual({})
    expect(objectArguments('["a","b"]')).toEqual({})
    expect(objectArguments('42')).toEqual({})
  })
})

describe('toObjectStyle', () => {
  it('keeps non-assistant messages and plain assistant text verbatim', () => {
    const wire = toObjectStyle([
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'go' },
      { role: 'assistant', content: 'done' },
      { role: 'tool', tool_call_id: 'call_1', content: 'ok' },
    ])
    expect(wire).toEqual([
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'go' },
      { role: 'assistant', content: 'done' },
      { role: 'tool', tool_call_id: 'call_1', content: 'ok' },
    ])
  })

  it('rewrites tool-call arguments to objects', () => {
    const wire = toObjectStyle([
      { role: 'assistant', content: '', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'bash', arguments: '{"command":"ls"}' } }] },
    ])
    expect(wire).toEqual([
      { role: 'assistant', content: '', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'bash', arguments: { command: 'ls' } } }] },
    ])
  })
})

describe('serializeRequest', () => {
  it('serializes a full tool conversation with object-style arguments', () => {
    const body = serializeRequest(baseOptions([
      assistantToolCall('call_1', 'bash', '{"command":"echo hi"}'),
      toolResult('call_1', 'hi\n'),
      text('user', 'what did it print?'),
    ]))
    expect(body.model).toBe('qwen3.8-27b')
    expect(body.stream).toBe(true)
    expect(body.stream_options).toEqual({ include_usage: true })
    expect(body.messages).toEqual([
      { role: 'assistant', content: '', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'bash', arguments: { command: 'echo hi' } } }] },
      { role: 'tool', tool_call_id: 'call_1', content: 'hi\n' },
      { role: 'user', content: 'what did it print?' },
    ])
    expect('thinking' in body).toBe(false)
    expect('reasoning_effort' in body).toBe(false)
  })

  it('prepends the system slot and maps tools with optionality', () => {
    const withAll = serializeRequest(
      { ...baseOptions([]), system: 'be brief', tools: [tool], temperature: 0.7, maxTokens: 512, stop: ['END'] },
    )
    expect(withAll.messages[0]).toEqual({ role: 'system', content: 'be brief' })
    expect(withAll.tools).toEqual([{
      type: 'function',
      function: { name: 'bash', description: 'Run a shell command', parameters: tool.parameters },
    }])
    expect(withAll.temperature).toBe(0.7)
    expect(withAll.max_tokens).toBe(512)
    expect(withAll.stop).toEqual(['END'])

    const bare = serializeRequest({ ...baseOptions([]), tools: [] })
    expect('tools' in bare).toBe(false)
    expect('temperature' in bare).toBe(false)
    expect('max_tokens' in bare).toBe(false)
    expect('stop' in bare).toBe(false)
    expect(bare.messages).toEqual([])
  })

  it('merges requestExtras under core fields and preserves gateway keys', () => {
    const body = serializeRequest(
      { ...baseOptions([]), temperature: 0.1 },
      {
        chat_template_kwargs: { enable_thinking: false },
        temperature: 9.9,
        presence_penalty: 1.5,
      },
    )
    expect(body.chat_template_kwargs).toEqual({ enable_thinking: false })
    expect(body.presence_penalty).toBe(1.5)
    // Core request fields win same-key collisions with extras.
    expect(body.temperature).toBe(0.1)
  })
})
