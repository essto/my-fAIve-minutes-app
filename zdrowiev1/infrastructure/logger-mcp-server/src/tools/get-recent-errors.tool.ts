import type { McpToolResult } from '@monorepo/mcp-shared';
import { GetRecentErrorsInput } from '../schemas/logger.schemas.js';
import { LoggerService } from '../logger.service.js';

export async function handleGetRecentErrors(
  params: GetRecentErrorsInput,
  loggerService: LoggerService,
): Promise<McpToolResult> {
  const errors = await loggerService.getRecentErrors(params.limit, params.level);

  const formatted = errors
    .map((e) => `[${e.level}] ${e.message}\n${e.stackTrace ? '  ' + e.stackTrace : ''}`)
    .join('\n\n');
  const textContent = `# Ostatnie logi\nZnaleziono ${errors.length} wpisów.\n\n${formatted}`;

  return {
    content: [{ type: 'text', text: textContent }],
    structuredContent: errors,
  };
}
