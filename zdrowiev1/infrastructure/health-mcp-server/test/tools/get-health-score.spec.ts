import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleGetHealthScore } from '../../src/tools/get-health-score.tool.js';
import type { McpApiClient } from '@monorepo/mcp-shared';

import { AxiosError, AxiosHeaders } from 'axios';

describe('health_get_health_score tool handler', () => {
  let mockClient: McpApiClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
      post: vi.fn(),
    } as unknown as McpApiClient;
  });

  it('returns aggregated health score with breakdown', async () => {
    // GIVEN
    vi.mocked(mockClient.get).mockResolvedValue({
      score: 78,
      breakdown: {
        weight: 85,
        sleep: 70,
        activity: 80,
      },
    });

    // WHEN
    const result = await handleGetHealthScore({ user_id: 'user-id' }, mockClient);

    // THEN
    expect(result.content).toHaveLength(1);
    expect(result.content[0].text).toContain('78');
    expect(result.content[0].text).toContain('Breakdown');
    expect((result.structuredContent as any).score).toBe(78);
    expect(mockClient.get).toHaveBeenCalledWith('/api/health-score', { userId: 'user-id' });
  });

  it('handles user with no health data gracefully', async () => {
    // GIVEN
    const error = new AxiosError('Not Found', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 404,
      statusText: 'Not Found',
      data: {},
      headers: {},
      config: { headers: new AxiosHeaders() },
    });
    vi.mocked(mockClient.get).mockRejectedValue(error);

    // WHEN
    const result = await handleGetHealthScore({ user_id: 'user-id' }, mockClient);

    // THEN
    expect(result.content[0].text).toContain('Resource not found');
  });
});
