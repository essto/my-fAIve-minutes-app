import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleSubmitSymptoms } from '../../src/tools/submit-symptoms.tool.js';
import type { McpApiClient } from '@monorepo/mcp-shared';

describe('diagnosis_submit_symptoms tool handler', () => {
  let mockClient: McpApiClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
      post: vi.fn(),
    } as unknown as McpApiClient;
  });

  it('submits symptoms and returns analysis', async () => {
    vi.mocked(mockClient.post).mockResolvedValue({
      id: 'report-123',
      userId: 'user-id',
      symptoms: ['ból głowy', 'gorączka'],
      triageLevel: 'YELLOW',
      recommendation: 'Skonsultuj się z lekarzem w ciągu 24h.',
    });

    const result = await handleSubmitSymptoms(
      {
        user_id: 'user-id',
        symptoms: ['ból głowy', 'gorączka'],
      },
      mockClient,
    );

    expect(result.content).toHaveLength(1);
    expect(result.content[0].text).toContain('zgłoszone i przeanalizowane');
    expect(result.content[0].text).toContain('YELLOW');
    expect((result.structuredContent as any).id).toBe('report-123');
    expect(mockClient.post).toHaveBeenCalledWith('/api/diagnosis/symptoms', {
      userId: 'user-id',
      symptoms: ['ból głowy', 'gorączka'],
    });
  });

  it('handles API errors gracefully', async () => {
    vi.mocked(mockClient.post).mockRejectedValue(new Error('Internal error'));

    const result = await handleSubmitSymptoms(
      {
        user_id: 'user-id',
        symptoms: ['kaszel'],
      },
      mockClient,
    );

    expect(result.content[0].text).toContain('Error');
  });
});
