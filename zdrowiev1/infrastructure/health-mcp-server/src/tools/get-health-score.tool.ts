import { McpApiClient, handleMcpError } from '@monorepo/mcp-shared';
import type { McpToolResult } from '@monorepo/mcp-shared';
import { GetHealthScoreInput } from '../schemas/health.schemas.js';

type HealthScoreResponse = {
  score: number;
  breakdown: {
    weight: number;
    sleep: number;
    activity: number;
  };
};

export async function handleGetHealthScore(
  params: GetHealthScoreInput,
  apiClient: McpApiClient,
): Promise<McpToolResult> {
  try {
    const data = await apiClient.get<HealthScoreResponse>('/api/health-score', {
      userId: params.user_id,
    });

    const textContent = `# Wynik Zdrowotny\nOgólny wynik wspierany algorytmem: **${data.score} / 100**\n\n## Breakdown\n- Waga: ${data.breakdown.weight}\n- Sen: ${data.breakdown.sleep}\n- Aktywność: ${data.breakdown.activity}`;

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
