import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleGetWeightHistory } from '../../src/tools/get-weight-history.tool';
import type { McpApiClient } from '@monorepo/mcp-shared';

describe('health_get_weight_history tool handler', () => {
  let mockClient: McpApiClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
      post: vi.fn(),
    } as unknown as McpApiClient;
  });

  it('returns formatted weight history for valid user_id', async () => {
    // GIVEN
    vi.mocked(mockClient.get).mockResolvedValue({
      total: 3,
      count: 3,
      offset: 0,
      has_more: false,
      items: [
        { value: 70, unit: 'kg', timestamp: '2023-01-01T10:00:00Z' },
        { value: 71, unit: 'kg', timestamp: '2023-01-08T10:00:00Z' },
        { value: 72, unit: 'kg', timestamp: '2023-01-15T10:00:00Z' },
      ],
    });

    // WHEN
    const result = await handleGetWeightHistory(
      { user_id: '123e4567-e89b-12d3-a456-426614174000', limit: 10, offset: 0 },
      mockClient,
    );

    // THEN
    expect(result.content).toHaveLength(1);
    expect(result.content[0].text).toContain('70');
    expect(result.content[0].text).toContain('72');
    expect((result.structuredContent as any).items).toHaveLength(3);
    expect(mockClient.get).toHaveBeenCalledWith('/api/weight', {
      userId: '123e4567-e89b-12d3-a456-426614174000',
      limit: 10,
      offset: 0,
    });
  });

  it('returns empty message when no readings found', async () => {
    // GIVEN
    vi.mocked(mockClient.get).mockResolvedValue({
      total: 0,
      count: 0,
      offset: 0,
      has_more: false,
      items: [],
    });

    // WHEN
    const result = await handleGetWeightHistory(
      { user_id: '123e4567-e89b-12d3-a456-426614174000', limit: 10, offset: 0 },
      mockClient,
    );

    // THEN
    expect(result.content[0].text).toContain('Brak odczytów');
    expect((result.structuredContent as any).items).toHaveLength(0);
  });

  it('handles API error gracefully', async () => {
    // GIVEN
    vi.mocked(mockClient.get).mockRejectedValue(new Error('Connection refused'));

    // WHEN
    const result = await handleGetWeightHistory(
      { user_id: '123e4567-e89b-12d3-a456-426614174000', limit: 20, offset: 0 },
      mockClient,
    );

    // THEN
    expect(result.content[0].text).toContain('Error');
  });
});
