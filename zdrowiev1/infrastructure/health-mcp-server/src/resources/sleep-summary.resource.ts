import { McpApiClient, handleMcpError } from '@monorepo/mcp-shared';

export async function handleSleepSummaryResource(userId: string, apiClient: McpApiClient) {
  try {
    const data = await apiClient.get<any>('/api/sleep/summary', { userId });

    const content = JSON.stringify(data, null, 2);

    return {
      contents: [
        {
          uri: `health://sleep/${userId}/summary`,
          mimeType: 'application/json',
          text: content,
        },
      ],
    };
  } catch (error) {
    throw new Error(handleMcpError(error));
  }
}
