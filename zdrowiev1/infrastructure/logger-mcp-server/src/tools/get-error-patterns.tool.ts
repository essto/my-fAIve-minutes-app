import type { McpToolResult } from '@monorepo/mcp-shared';
import { GetErrorPatternsInput } from '../schemas/logger.schemas.js';
import { LoggerService } from '../logger.service.js';

export async function handleGetErrorPatterns(
  params: GetErrorPatternsInput,
  loggerService: LoggerService,
): Promise<McpToolResult> {
  const patterns = await loggerService.getErrorPatterns(params.timeframe_hours);

  const textContent =
    `# Wzorce błędów (ostatnie ${params.timeframe_hours}h)\n` +
    patterns.map((p) => `- **${p.count}x:** ${p.pattern}`).join('\n');

  return {
    content: [{ type: 'text', text: textContent }],
    structuredContent: patterns,
  };
}
