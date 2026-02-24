#!/usr/bin/env node

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

import {
  GetRecentErrorsInputSchema,
  GetErrorPatternsInputSchema,
  SuggestFixInputSchema,
} from './schemas/logger.schemas.js';

import { handleGetRecentErrors } from './tools/get-recent-errors.tool.js';
import { handleGetErrorPatterns } from './tools/get-error-patterns.tool.js';
import { handleSuggestFix } from './tools/suggest-fix.tool.js';
import { LoggerService } from './logger.service.js';

const server = new McpServer({
  name: 'logger-mcp-server',
  version: '1.0.0',
});

const loggerService = new LoggerService();

// 1. Tool: logger_get_recent_errors
server.registerTool(
  'logger_get_recent_errors',
  {
    title: 'Get Recent Errors',
    description: 'Pobiera ostatnie błędy z logów.',
    inputSchema: GetRecentErrorsInputSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },
  async (params: any) => (await handleGetRecentErrors(params, loggerService)) as any,
);

// 2. Tool: logger_get_error_patterns
server.registerTool(
  'logger_get_error_patterns',
  {
    title: 'Get Error Patterns',
    description: 'Pobiera często powtarzające się wzorce błędów.',
    inputSchema: GetErrorPatternsInputSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },
  async (params: any) => (await handleGetErrorPatterns(params, loggerService)) as any,
);

// 3. Tool: logger_suggest_fix
server.registerTool(
  'logger_suggest_fix',
  {
    title: 'Suggest Fix',
    description: 'Zwraca sugestię naprawy błędu na podstawie wzorca.',
    inputSchema: SuggestFixInputSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  async (params: any) => (await handleSuggestFix(params, loggerService)) as any,
);

async function run() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('MCP server "logger-mcp-server" running via stdio');
}

run().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
