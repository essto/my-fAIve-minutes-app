import { McpApiClient, handleMcpError, formatMarkdownTable } from '@monorepo/mcp-shared';
import type { McpToolResult } from '@monorepo/mcp-shared';
import { SearchFoodInput } from '../schemas/diet.schemas.js';

export async function handleSearchFood(
  params: SearchFoodInput,
  apiClient: McpApiClient,
): Promise<McpToolResult> {
  try {
    const data = await apiClient.get<any[]>('/api/diet/food/search', {
      q: params.query,
    });

    if (!data || data.length === 0) {
      return {
        content: [{ type: 'text', text: `Brak wyników dla: ${params.query}` }],
        structuredContent: { items: data },
      };
    }

    const headers = ['Nazwa', 'Kcal', 'Białko', 'Węgle', 'Tłuszcze'];
    const rows = data.map((item) => [
      item.name || '-',
      item.calories?.toString() || '0',
      item.protein?.toString() || '0',
      item.carbs?.toString() || '0',
      item.fat?.toString() || '0',
    ]);

    const table = formatMarkdownTable(headers, rows);
    const textContent = `# Wyniki wyszukiwania dla: ${params.query}\nZnaleziono ${data.length} produktów.\n\n${table}`;

    return {
      content: [{ type: 'text', text: textContent }],
      structuredContent: { items: data },
    };
  } catch (error) {
    return {
      content: [{ type: 'text', text: handleMcpError(error) }],
    };
  }
}
