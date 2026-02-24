#!/usr/bin/env node

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { McpApiClient } from '@monorepo/mcp-shared';

import { setupServer } from './server-setup.js';

const server = new McpServer({
  name: 'health-mcp-server',
  version: '1.0.0',
});

const apiClient = new McpApiClient();
const transport = new StdioServerTransport();

setupServer(server, apiClient);

async function run() {
  await server.connect(transport);
  console.error('MCP server "health-mcp-server" running via stdio');
}

// Graceful shutdown handling
const shutdown = async () => {
  console.error('\nShutting down health-mcp-server...');
  try {
    await server.close();
    process.exit(0);
  } catch (err) {
    console.error('Error during shutdown:', err);
    process.exit(1);
  }
};

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

run().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
