import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleGetTriage } from '../../src/tools/get-triage.tool.js';
import type { McpApiClient } from '@monorepo/mcp-shared';
import { AxiosError, AxiosHeaders } from 'axios';

describe('diagnosis_get_triage tool handler', () => {
  let mockClient: McpApiClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
      post: vi.fn(),
    } as unknown as McpApiClient;
  });

  it('returns triage info', async () => {
    vi.mocked(mockClient.get).mockResolvedValue({
      reportId: 'report-id',
      triageLevel: 'RED',
      recommendation: 'Natychmiast wezwij pogotowie.',
    });

    const result = await handleGetTriage({ report_id: 'report-id' }, mockClient);

    expect(result.content).toHaveLength(1);
    expect(result.content[0].text).toContain('RED');
    expect(result.content[0].text).toContain('Natychmiast wezwij pogotowie');
    expect((result.structuredContent as any).triageLevel).toBe('RED');
    expect(mockClient.get).toHaveBeenCalledWith('/api/diagnosis/triage', { reportId: 'report-id' });
  });

  it('handles API errors', async () => {
    const error = new AxiosError('Not Found', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 404,
      statusText: 'Not Found',
      data: {},
      headers: {},
      config: { headers: new AxiosHeaders() },
    });
    vi.mocked(mockClient.get).mockRejectedValue(error);

    const result = await handleGetTriage({ report_id: 'nonexistent' }, mockClient);

    expect(result.content[0].text).toContain('Resource not found');
  });
});
