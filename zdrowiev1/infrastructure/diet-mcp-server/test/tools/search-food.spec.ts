import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleSearchFood } from '../../src/tools/search-food.tool.js';
import type { McpApiClient } from '@monorepo/mcp-shared';

describe('diet_search_food tool handler', () => {
  let mockClient: McpApiClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
      post: vi.fn(),
    } as unknown as McpApiClient;
  });

  it('returns matching food items from database', async () => {
    vi.mocked(mockClient.get).mockResolvedValue([
      { id: '1', name: 'Jabłko', calories: 52, protein: 0.3, carbs: 14, fat: 0.2 },
      { id: '2', name: 'Sok jabłkowy', calories: 45, protein: 0.1, carbs: 11, fat: 0.1 },
    ]);

    const result = await handleSearchFood({ query: 'jabłko' }, mockClient);

    expect(result.content).toHaveLength(1);
    expect(result.content[0].text).toContain('Wyniki wyszukiwania dla: jabłko');
    expect(result.content[0].text).toContain('Jabłko');
    expect((result.structuredContent as any).items).toHaveLength(2);
    expect(mockClient.get).toHaveBeenCalledWith('/api/diet/food/search', { q: 'jabłko' });
  });

  it('handles empty results', async () => {
    vi.mocked(mockClient.get).mockResolvedValue([]);

    const result = await handleSearchFood({ query: 'nonexistent' }, mockClient);

    expect(result.content[0].text).toContain('Brak wyników');
  });

  it('handles API errors', async () => {
    vi.mocked(mockClient.get).mockRejectedValue(new Error('API Down'));

    const result = await handleSearchFood({ query: 'test' }, mockClient);

    expect(result.content[0].text).toContain('Error');
  });
});
