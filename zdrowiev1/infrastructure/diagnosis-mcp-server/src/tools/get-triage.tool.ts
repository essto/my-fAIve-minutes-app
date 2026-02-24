import { McpApiClient, handleMcpError } from '@monorepo/mcp-shared';
import type { McpToolResult } from '@monorepo/mcp-shared';
import { GetTriageInput } from '../schemas/diagnosis.schemas.js';

export async function handleGetTriage(
  params: GetTriageInput,
  apiClient: McpApiClient,
): Promise<McpToolResult> {
  try {
    const data = await apiClient.get<any>('/api/diagnosis/triage', {
      reportId: params.report_id,
    });

    const textContent = `# Wyniki Triage\n**Raport ID:** ${params.report_id}\n**Poziom zagrożenia:** ${data.triageLevel}\n**Zalecenie:** ${data.recommendation}`;

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
