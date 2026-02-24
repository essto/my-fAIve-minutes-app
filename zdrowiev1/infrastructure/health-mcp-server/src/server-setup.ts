import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { McpApiClient } from '@monorepo/mcp-shared';
import { z } from 'zod';

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

export function setupServer(server: McpServer, apiClient: McpApiClient) {
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
        async (params: z.infer<typeof GetWeightHistoryInputSchema>) => {
            try {
                const res = await handleGetWeightHistory(params, apiClient);
                return res as any;
            } catch (error: any) {
                return {
                    content: [{ type: 'text', text: `Error fetching weight history: ${error?.message}` }],
                    isError: true,
                } as any;
            }
        }
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
        async (params: z.infer<typeof AddWeightReadingInputSchema>) => {
            try {
                const res = await handleAddWeightReading(params, apiClient);
                return res as any;
            } catch (error: any) {
                return {
                    content: [{ type: 'text', text: `Error adding weight reading: ${error?.message}` }],
                    isError: true,
                } as any;
            }
        }
    );

    // 3. Tool: get_health_score
    server.registerTool(
        'health_get_health_score',
        {
            title: 'Get Health Score',
            description: 'Pobiera ogólny wynik zdrowotny (Health Score) na podstawie danych z różnych modułów.',
            inputSchema: GetHealthScoreInputSchema,
            annotations: {
                readOnlyHint: true,
                destructiveHint: false,
                idempotentHint: true,
                openWorldHint: true,
            },
        },
        async (params: z.infer<typeof GetHealthScoreInputSchema>) => {
            try {
                const res = await handleGetHealthScore(params, apiClient);
                return res as any;
            } catch (error: any) {
                return {
                    content: [{ type: 'text', text: `Error getting health score: ${error?.message}` }],
                    isError: true,
                } as any;
            }
        }
    );

    // Resources
    server.registerResource(
        'health://weight/{userId}/trend',
        'Weight Trend Analysis',
        {
            description: 'Analiza trendów wagi użytkownika',
            mimeType: 'application/json',
        },
        async (uri: any) => {
            try {
                const match = uri.href.match(/^health:\/\/weight\/([^/]+)\/trend$/);
                if (!match) throw new Error('Invalid URI');
                const res = await handleWeightTrendResource(match[1], apiClient);
                return res as any;
            } catch (error: any) {
                return {
                    contents: [{ uri: uri.href, mimeType: 'application/json', text: JSON.stringify({ error: error.message }) }]
                } as any;
            }
        }
    );

    server.registerResource(
        'health://sleep/{userId}/summary',
        'Sleep Summary',
        {
            description: 'Podsumowanie snu użytkownika',
            mimeType: 'application/json',
        },
        async (uri: any) => {
            try {
                const match = uri.href.match(/^health:\/\/sleep\/([^/]+)\/summary$/);
                if (!match) throw new Error('Invalid URI');
                const res = await handleSleepSummaryResource(match[1], apiClient);
                return res as any;
            } catch (error: any) {
                return {
                    contents: [{ uri: uri.href, mimeType: 'application/json', text: JSON.stringify({ error: error.message }) }]
                } as any;
            }
        }
    );
}
