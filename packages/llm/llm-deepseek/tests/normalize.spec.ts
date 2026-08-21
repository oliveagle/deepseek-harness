import { describe, expect, it } from 'vitest'
import { normalizeToolCallIdentity } from '../src/serialize.ts'
import type { WireMessage } from '../src/types.ts'

function assistant(ids: Array<string | undefined>): WireMessage {
  return {
    role: 'assistant',
    content: '',
    tool_calls: ids.map(id => ({ id: id ?? '', type: 'function' as const, function: { name: 'read', arguments: '{}' } })),
  }
}
function tool(id: string): WireMessage {
  return { role: 'tool', tool_call_id: id, content: 'out' }
}

describe('normalizeToolCallIdentity (session-poison self-heal)', () => {
  it('passes well-formed foreign-style (bare UUID) ids verbatim, pairs stay matched', () => {
    const wire = normalizeToolCallIdentity([
      assistant(['b99f439c-c5c2-4c14-ac36-0bda43cab5e5']),
      tool('b99f439c-c5c2-4c14-ac36-0bda43cab5e5'),
    ])
    const call = (wire[0] as { tool_calls: Array<{ id: string }> }).tool_calls[0]!.id
    expect(call).toBe('b99f439c-c5c2-4c14-ac36-0bda43cab5e5')
    expect((wire[1] as { tool_call_id: string }).tool_call_id).toBe(call)
  })

  it('dedupes a repeated id and remaps the second pair', () => {
    const wire = normalizeToolCallIdentity([
      assistant(['call_e4605f6459374c52b3fb7277']),
      tool('call_e4605f6459374c52b3fb7277'),
      assistant(['call_e4605f6459374c52b3fb7277']),
      tool('call_e4605f6459374c52b3fb7277'),
    ])
    const first = (wire[0] as { tool_calls: Array<{ id: string }> }).tool_calls[0]!.id
    const second = (wire[2] as { tool_calls: Array<{ id: string }> }).tool_calls[0]!.id
    expect(first).not.toBe(second)
    expect((wire[1] as { tool_call_id: string }).tool_call_id).toBe(first)
    expect((wire[3] as { tool_call_id: string }).tool_call_id).toBe(second)
  })

  it('synthesizes an id for empty tool_call ids', () => {
    const wire = normalizeToolCallIdentity([assistant([''])])
    const call = (wire[0] as { tool_calls: Array<{ id: string }> }).tool_calls[0]!.id
    expect(call).toMatch(/^call_[0-9a-f]{32}$/)
  })

  it('leaves already-canonical unique histories untouched', () => {
    const input = [assistant(['call_0123456789abcdef']), tool('call_0123456789abcdef')]
    const before = JSON.stringify(input)
    expect(JSON.stringify(normalizeToolCallIdentity(input))).toBe(before)
  })
})
