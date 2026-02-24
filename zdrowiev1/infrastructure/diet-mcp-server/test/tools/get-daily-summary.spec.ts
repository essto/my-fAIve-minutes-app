import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleGetDailySummary } from '../../src/tools/get-daily-summary.tool.js';
import type { McpApiClient } from '@monorepo/mcp-shared';
import { AxiosError, AxiosHeaders } from 'axios';

describe('diet_get_daily_summary tool handler', () => {
  let mockClient: McpApiClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
      post: vi.fn(),
    } as unknown as McpApiClient;
  });

  it('returns daily summary with meals', async () => {
    vi.mocked(mockClient.get).mockResolvedValue({
      totalCalories: 2000,
      totalProtein: 150,
      totalCarbs: 200,
      totalFat: 60,
      meals: [
        { name: 'Owsianka', calories: 350 },
        { name: 'Kurczak z ryżem', calories: 650 },
      ],
    });

    const result = await handleGetDailySummary(
      { user_id: 'user-id', date: '2023-05-15' },
      mockClient,
    );

    expect(result.content).toHaveLength(1);
    expect(result.content[0].text).toContain('Podsumowanie dna: 2023-05-15');
    expect(result.content[0].text).toContain('2000 kcal');
    expect((result.structuredContent as any).meals).toHaveLength(2);
    expect(mockClient.get).toHaveBeenCalledWith('/api/diet/summary', {
      userId: 'user-id',
      date: '2023-05-15',
    });
  });

  it('handles empty day gracefully', async () => {
    vi.mocked(mockClient.get).mockResolvedValue({
      totalCalories: 0,
      totalProtein: 0,
      totalCarbs: 0,
      totalFat: 0,
      meals: [],
    });

    const result = await handleGetDailySummary(
      { user_id: 'user-id', date: '2023-05-15' },
      mockClient,
    );

    expect(result.content[0].text).toContain('Brak zarejestrowanych posiłków');
  });

  it('handles API errors gracefully', async () => {
    const error = new AxiosError('Not Found', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 404,
      statusText: 'Not Found',
      data: {},
      headers: {},
      config: { headers: new AxiosHeaders() },
    });
    vi.mocked(mockClient.get).mockRejectedValue(error);

    const result = await handleGetDailySummary(
      { user_id: 'user-id', date: '2023-05-15' },
      mockClient,
    );

    expect(result.content[0].text).toContain('Resource not found');
  });
});
