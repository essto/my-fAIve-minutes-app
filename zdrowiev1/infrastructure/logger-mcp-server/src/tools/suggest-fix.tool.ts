import type { McpToolResult } from '@monorepo/mcp-shared';
import { SuggestFixInput } from '../schemas/logger.schemas.js';
import { LoggerService } from '../logger.service.js';

export async function handleSuggestFix(
  params: SuggestFixInput,
  loggerService: LoggerService,
): Promise<McpToolResult> {
  const suggestion = await loggerService.suggestFix(params.error_pattern);

  return {
    content: [{ type: 'text', text: suggestion }],
    structuredContent: { suggestion },
  };
}
