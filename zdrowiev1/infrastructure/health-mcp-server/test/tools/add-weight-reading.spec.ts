import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleAddWeightReading } from '../../src/tools/add-weight-reading.tool.js';
import type { McpApiClient } from '@monorepo/mcp-shared';

describe('health_add_weight_reading tool handler', () => {
  let mockClient: McpApiClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
      post: vi.fn(),
    } as unknown as McpApiClient;
  });

  it('saves valid weight reading and returns confirmation', async () => {
    // GIVEN
    vi.mocked(mockClient.post).mockResolvedValue({
      id: 'reading-123',
      userId: 'user-id',
      value: 75.5,
      unit: 'kg',
      timestamp: new Date().toISOString(),
    });

    // WHEN
    const result = await handleAddWeightReading(
      {
        user_id: 'user-id',
        value: 75.5,
        unit: 'kg',
      },
      mockClient,
    );

    // THEN
    expect(result.content).toHaveLength(1);
    expect(result.content[0].text).toContain('Pomiar wagi został dodany');
    expect((result.structuredContent as any).value).toBe(75.5);
    expect(mockClient.post).toHaveBeenCalledWith('/api/weight', {
      userId: 'user-id',
      value: 75.5,
      unit: 'kg',
    });
  });

  it('handles API error gracefully', async () => {
    // GIVEN
    vi.mocked(mockClient.post).mockRejectedValue(new Error('Internal server error'));

    // WHEN
    const result = await handleAddWeightReading(
      {
        user_id: 'user-id',
        value: 75.5,
        unit: 'kg',
      },
      mockClient,
    );

    // THEN
    expect(result.content[0].text).toContain('Error');
  });
});
