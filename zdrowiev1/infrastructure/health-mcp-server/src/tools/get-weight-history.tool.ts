import { McpApiClient, handleMcpError, formatMarkdownTable } from '@monorepo/mcp-shared';
import type { McpToolResult, PaginatedResponse } from '@monorepo/mcp-shared';
import { GetWeightHistoryInput } from '../schemas/health.schemas.js';

type WeightReadingResponse = {
  value: number;
  unit: string;
  timestamp: string;
};

export async function handleGetWeightHistory(
  params: GetWeightHistoryInput,
  apiClient: McpApiClient,
): Promise<McpToolResult> {
  try {
    const data = await apiClient.get<PaginatedResponse<WeightReadingResponse>>(`/api/weight`, {
      userId: params.user_id,
      limit: params.limit,
      offset: params.offset,
    });

    if (!data.items || data.items.length === 0) {
      return {
        content: [{ type: 'text', text: 'Brak odczytów wagi dla tego użytkownika.' }],
        structuredContent: data,
      };
    }

    const headers = ['Data', 'Waga', 'Jednostka'];
    const rows = data.items.map((item) => [
      new Date(item.timestamp).toLocaleDateString('pl-PL', { hour: '2-digit', minute: '2-digit' }),
      item.value.toString(),
      item.unit,
    ]);

    const table = formatMarkdownTable(headers, rows);
    const textContent = `# Historia Wagi\n\nZnaleziono ${data.total} wpisów (pokazuję ${data.count}).\n\n${table}`;

    return {
      content: [{ type: 'text', text: textContent }],
      structuredContent: data,
    };
  } catch (error) {
    return {
      content: [{ type: 'text', text: handleMcpError(error) }],
    };
  }
}
