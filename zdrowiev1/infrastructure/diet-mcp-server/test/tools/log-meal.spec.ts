import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleLogMeal } from '../../src/tools/log-meal.tool.js';
import type { McpApiClient } from '@monorepo/mcp-shared';

describe('diet_log_meal tool handler', () => {
  let mockClient: McpApiClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
      post: vi.fn(),
    } as unknown as McpApiClient;
  });

  it('logs meal successfully and returns confirmation', async () => {
    vi.mocked(mockClient.post).mockResolvedValue({
      id: 'meal-id',
      userId: 'user-id',
      name: 'Owsianka',
      calories: 350,
      protein: 10,
      carbs: 60,
      fat: 5,
      createdAt: new Date().toISOString(),
    });

    const result = await handleLogMeal(
      {
        user_id: 'user-id',
        name: 'Owsianka',
        calories: 350,
        protein: 10,
      },
      mockClient,
    );

    expect(result.content).toHaveLength(1);
    expect(result.content[0].text).toContain('Posiłek zapisany');
    expect(result.content[0].text).toContain('350 kcal');
    expect(mockClient.post).toHaveBeenCalledWith('/api/diet/meals', {
      userId: 'user-id',
      name: 'Owsianka',
      calories: 350,
      protein: 10,
    });
  });

  it('handles API errors', async () => {
    vi.mocked(mockClient.post).mockRejectedValue(new Error('Failed to connect'));
    const result = await handleLogMeal(
      { user_id: 'user-id', name: 'Jabłko', calories: 50 },
      mockClient,
    );
    expect(result.content[0].text).toContain('Error');
  });
});
