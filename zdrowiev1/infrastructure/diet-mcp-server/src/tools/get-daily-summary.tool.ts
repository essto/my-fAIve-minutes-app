import { McpApiClient, handleMcpError, formatMarkdownTable } from '@monorepo/mcp-shared';
import type { McpToolResult } from '@monorepo/mcp-shared';
import { GetDailySummaryInput } from '../schemas/diet.schemas.js';

export async function handleGetDailySummary(
  params: GetDailySummaryInput,
  apiClient: McpApiClient,
): Promise<McpToolResult> {
  try {
    const data = await apiClient.get<any>('/api/diet/summary', {
      userId: params.user_id,
      date: params.date,
    });

    if (data.meals && data.meals.length === 0) {
      return {
        content: [
          { type: 'text', text: `Brak zarejestrowanych posiłków dla daty ${params.date}.` },
        ],
        structuredContent: data,
      };
    }

    const headers = ['Posiłek', 'Kcal', 'Białko', 'Węgle', 'Tłuszcze'];
    const rows = (data.meals || []).map((meal: any) => [
      meal.name || '-',
      meal.calories?.toString() || '0',
      meal.protein?.toString() || '0',
      meal.carbs?.toString() || '0',
      meal.fat?.toString() || '0',
    ]);

    const table = formatMarkdownTable(headers, rows);
    const textContent = `# Podsumowanie dna: ${params.date}\n- **Kalorie:** ${data.totalCalories} kcal\n- **Białko:** ${data.totalProtein}g\n- **Węglowodany:** ${data.totalCarbs}g\n- **Tłuszcze:** ${data.totalFat}g\n\n## Posiłki\n${table}`;

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
