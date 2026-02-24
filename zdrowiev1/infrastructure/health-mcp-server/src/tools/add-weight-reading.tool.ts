import { McpApiClient, handleMcpError } from '@monorepo/mcp-shared';
import type { McpToolResult } from '@monorepo/mcp-shared';
import { AddWeightReadingInput } from '../schemas/health.schemas.js';

export async function handleAddWeightReading(
  params: AddWeightReadingInput,
  apiClient: McpApiClient,
): Promise<McpToolResult> {
  try {
    const data = await apiClient.post<any>('/api/weight', {
      userId: params.user_id,
      value: params.value,
      unit: params.unit,
    });

    const textContent = `# Pomiar zapisany\nPomiar wagi został dodany pomyślnie. Nowy odczyt: ${data.value} ${data.unit}.`;

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
