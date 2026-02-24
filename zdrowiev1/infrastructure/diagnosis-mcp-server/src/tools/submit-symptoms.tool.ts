import { McpApiClient, handleMcpError } from '@monorepo/mcp-shared';
import type { McpToolResult } from '@monorepo/mcp-shared';
import { SubmitSymptomsInput } from '../schemas/diagnosis.schemas.js';

export async function handleSubmitSymptoms(
  params: SubmitSymptomsInput,
  apiClient: McpApiClient,
): Promise<McpToolResult> {
  try {
    const data = await apiClient.post<any>('/api/diagnosis/symptoms', {
      userId: params.user_id,
      symptoms: params.symptoms,
    });

    const textContent = `# Objawy zgłoszone i przeanalizowane\n**Poziom Triage:** ${data.triageLevel}\n**Rekomendacja:** ${data.recommendation}\n\n*Identyfikator raportu: ${data.id} - użyj go w innych narzędziach.*`;

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
