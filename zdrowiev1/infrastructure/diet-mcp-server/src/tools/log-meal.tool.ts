import { McpApiClient, handleMcpError } from '@monorepo/mcp-shared';
import type { McpToolResult } from '@monorepo/mcp-shared';
import { LogMealInput } from '../schemas/diet.schemas.js';

export async function handleLogMeal(
  params: LogMealInput,
  apiClient: McpApiClient,
): Promise<McpToolResult> {
  try {
    const payload: any = {
      userId: params.user_id,
      name: params.name,
      calories: params.calories,
    };
    if (params.protein !== undefined) payload.protein = params.protein;
    if (params.carbs !== undefined) payload.carbs = params.carbs;
    if (params.fat !== undefined) payload.fat = params.fat;

    const data = await apiClient.post<any>('/api/diet/meals', payload);

    return {
      content: [
        {
          type: 'text',
          text: `# Posiłek zapisany\nPomyślnie dodano: **${data.name}** (${data.calories} kcal).`,
        },
      ],
      structuredContent: data,
    };
  } catch (error) {
    return { content: [{ type: 'text', text: handleMcpError(error) }] };
  }
}
