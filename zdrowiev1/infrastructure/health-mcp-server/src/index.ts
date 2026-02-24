#!/usr/bin/env node

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { McpApiClient } from '@monorepo/mcp-shared';

import {
  GetWeightHistoryInputSchema,
  AddWeightReadingInputSchema,
  GetHealthScoreInputSchema,
} from './schemas/health.schemas.js';

import { handleGetWeightHistory } from './tools/get-weight-history.tool.js';
import { handleAddWeightReading } from './tools/add-weight-reading.tool.js';
import { handleGetHealthScore } from './tools/get-health-score.tool.js';

import { handleWeightTrendResource } from './resources/weight-trend.resource.js';
import { handleSleepSummaryResource } from './resources/sleep-summary.resource.js';

const server = new McpServer({
  name: 'health-mcp-server',
  version: '1.0.0',
});

const apiClient = new McpApiClient();

// 1. Tool: get_weight_history
server.registerTool(
  'health_get_weight_history',
  {
    title: 'Get Weight History',
    description: 'Pobiera historię pomiarów wagi użytkownika z opcjonalną paginacją.',
    inputSchema: GetWeightHistoryInputSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },
  async (params) => handleGetWeightHistory(params, apiClient),
);

// 2. Tool: add_weight_reading
server.registerTool(
  'health_add_weight_reading',
  {
    title: 'Add Weight Reading',
    description: 'Dodaje nowy pomiar wagi użytkownika.',
    inputSchema: AddWeightReadingInputSchema,
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  async (params) => handleAddWeightReading(params, apiClient),
);

// 3. Tool: get_health_score
server.registerTool(
  'health_get_health_score',
  {
    title: 'Get Health Score',
    description:
      'Pobiera ogólny wynik zdrowotny (Health Score) na podstawie danych z różnych modułów.',
    inputSchema: GetHealthScoreInputSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },
  async (params) => handleGetHealthScore(params, apiClient),
);

// Resources
server.registerResource(
  {
    uri: 'health://weight/{userId}/trend',
    name: 'Weight Trend Analysis',
    description: 'Analiza trendów wagi użytkownika',
    mimeType: 'application/json',
  },
  async (uri) => {
    const match = uri.match(/^health:\/\/weight\/([^/]+)\/trend$/);
    if (!match) throw new Error('Invalid URI');
    return handleWeightTrendResource(match[1], apiClient);
  },
);

server.registerResource(
  {
    uri: 'health://sleep/{userId}/summary',
    name: 'Sleep Summary',
    description: 'Podsumowanie snu użytkownika',
    mimeType: 'application/json',
  },
  async (uri) => {
    const match = uri.match(/^health:\/\/sleep\/([^/]+)\/summary$/);
    if (!match) throw new Error('Invalid URI');
    return handleSleepSummaryResource(match[1], apiClient);
  },
);

async function run() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('MCP server "health-mcp-server" running via stdio');
}

run().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
