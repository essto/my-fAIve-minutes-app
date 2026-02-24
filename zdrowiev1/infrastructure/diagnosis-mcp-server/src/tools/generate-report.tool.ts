import { McpApiClient, handleMcpError } from '@monorepo/mcp-shared';
import type { McpToolResult } from '@monorepo/mcp-shared';
import { GenerateReportInput } from '../schemas/diagnosis.schemas.js';

export async function handleGenerateReport(
  params: GenerateReportInput,
  apiClient: McpApiClient,
): Promise<McpToolResult> {
  try {
    // We send format as a query param
    const data = await apiClient.get<any>(
      `/api/diagnosis/report/${params.report_id}?format=${params.format}`,
    );

    const textContent =
      params.format === 'pdf'
        ? `# Raport pobrany jako PDF\nLink do pobrania dokumentu: ${data.url || 'Brak URL'}`
        : `# Raport JSON\n${JSON.stringify(data, null, 2)}`;

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
