#!/usr/bin/env node

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { McpApiClient } from '@monorepo/mcp-shared';

import {
  LogMealInputSchema,
  GetDailySummaryInputSchema,
  SearchFoodInputSchema,
} from './schemas/diet.schemas.js';

import { handleLogMeal } from './tools/log-meal.tool.js';
import { handleGetDailySummary } from './tools/get-daily-summary.tool.js';
import { handleSearchFood } from './tools/search-food.tool.js';

const server = new McpServer({
  name: 'diet-mcp-server',
  version: '1.0.0',
});

const apiClient = new McpApiClient();

// 1. Tool: diet_log_meal
server.registerTool(
  'diet_log_meal',
  {
    title: 'Log Meal',
    description: 'Rejestruje posiłek dla użytkownika.',
    inputSchema: LogMealInputSchema,
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  async (params: any) => (await handleLogMeal(params, apiClient)) as any,
);

// 2. Tool: diet_get_daily_summary
server.registerTool(
  'diet_get_daily_summary',
  {
    title: 'Get Daily Summary',
    description: 'Pobiera dzienne podsumowanie makroskładników i kalorii.',
    inputSchema: GetDailySummaryInputSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },
  async (params: any) => (await handleGetDailySummary(params, apiClient)) as any,
);

// 3. Tool: diet_search_food
server.registerTool(
  'diet_search_food',
  {
    title: 'Search Food',
    description: 'Wyszukuje produkty z tabeli nutrition_data.',
    inputSchema: SearchFoodInputSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },
  async (params: any) => (await handleSearchFood(params, apiClient)) as any,
);

async function run() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('MCP server "diet-mcp-server" running via stdio');
}

run().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
