#!/usr/bin/env node

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { McpApiClient } from '@monorepo/mcp-shared';

import {
  SubmitSymptomsInputSchema,
  GetTriageInputSchema,
  GenerateReportInputSchema,
} from './schemas/diagnosis.schemas.js';

import { handleSubmitSymptoms } from './tools/submit-symptoms.tool.js';
import { handleGetTriage } from './tools/get-triage.tool.js';
import { handleGenerateReport } from './tools/generate-report.tool.js';

const server = new McpServer({
  name: 'diagnosis-mcp-server',
  version: '1.0.0',
});

const apiClient = new McpApiClient();

// 1. Tool: diagnosis_submit_symptoms
server.registerTool(
  'diagnosis_submit_symptoms',
  {
    title: 'Submit Symptoms',
    description: 'Zgłasza nowe objawy dla użytkownika i generuje triage.',
    inputSchema: SubmitSymptomsInputSchema,
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  async (params: any) => (await handleSubmitSymptoms(params, apiClient)) as any,
);

// 2. Tool: diagnosis_get_triage
server.registerTool(
  'diagnosis_get_triage',
  {
    title: 'Get Triage INFO',
    description: 'Pobiera poziom zagrożenia dla zgłoszonych objawów.',
    inputSchema: GetTriageInputSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },
  async (params: any) => (await handleGetTriage(params, apiClient)) as any,
);

// 3. Tool: diagnosis_generate_report
server.registerTool(
  'diagnosis_generate_report',
  {
    title: 'Generate Medical Report',
    description: 'Generuje oficjalny medyczny raport z triagu (json lub pdf).',
    inputSchema: GenerateReportInputSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },
  async (params: any) => (await handleGenerateReport(params, apiClient)) as any,
);

async function run() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('MCP server "diagnosis-mcp-server" running via stdio');
}

run().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
