import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleGenerateReport } from '../../src/tools/generate-report.tool.js';
import type { McpApiClient } from '@monorepo/mcp-shared';

describe('diagnosis_generate_report tool handler', () => {
  let mockClient: McpApiClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
      post: vi.fn(),
    } as unknown as McpApiClient;
  });

  it('returns JSON format successfully', async () => {
    vi.mocked(mockClient.get).mockResolvedValue({
      reportId: '123',
      content: 'Fake JSON Report Body',
      createdAt: '2023-01-01',
    });

    const result = await handleGenerateReport({ report_id: '123', format: 'json' }, mockClient);

    expect(result.content[0].text).toContain('Fake JSON Report Body');
    expect(mockClient.get).toHaveBeenCalledWith('/api/diagnosis/report/123?format=json');
  });

  it('returns PDF base64 note successfully', async () => {
    // For PDF, pretend the API returns some binary string / base64 or link
    vi.mocked(mockClient.get).mockResolvedValue({
      url: 'http://localhost:3001/api/diagnosis/report/123?format=pdf',
    });

    const result = await handleGenerateReport({ report_id: '123', format: 'pdf' }, mockClient);

    expect(result.content[0].text).toContain('http');
  });

  it('handles errors', async () => {
    vi.mocked(mockClient.get).mockRejectedValue(new Error('Internal Server Error'));
    const result = await handleGenerateReport({ report_id: '123', format: 'json' }, mockClient);
    expect(result.content[0].text).toContain('Error');
  });
});
