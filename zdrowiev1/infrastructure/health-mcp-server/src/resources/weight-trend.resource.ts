import { McpApiClient, handleMcpError } from '@monorepo/mcp-shared';

export async function handleWeightTrendResource(userId: string, apiClient: McpApiClient) {
  try {
    // We get history and analyze trend, simulating how a resource could aggregate it.
    // Assuming backend returns trend analysis or we calculate it here based on history.
    // For this example, we'll fetch history and just return a simulated trend based on history.
    const data = await apiClient.get<any>('/api/weight', { userId, limit: 100 });

    // Simplistic text result for demonstration, normally would call a trend API.
    const content = JSON.stringify(
      {
        userId,
        trend:
          data.items && data.items.length > 1 ? 'Calculated trend' : 'Not enough data for trend',
        dataPoints: data.count || 0,
      },
      null,
      2,
    );

    return {
      contents: [
        {
          uri: `health://weight/${userId}/trend`,
          mimeType: 'application/json',
          text: content,
        },
      ],
    };
  } catch (error) {
    throw new Error(handleMcpError(error));
  }
}
