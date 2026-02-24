import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios, { AxiosError, AxiosHeaders } from 'axios';
import { McpApiClient, handleMcpError, formatMarkdownTable } from '../src/api-client';

const { mockGet, mockPost } = vi.hoisted(() => ({
  mockGet: vi.fn(),
  mockPost: vi.fn(),
}));

vi.mock('axios', async (importOriginal) => {
  const actual = await importOriginal<typeof import('axios')>();
  return {
    ...actual,
    default: {
      create: vi.fn(() => ({
        get: mockGet,
        post: mockPost,
      })),
    },
  };
});

describe('mcp-shared utilities', () => {
  describe('McpApiClient', () => {
    let client: McpApiClient;

    beforeEach(() => {
      vi.clearAllMocks();
      client = new McpApiClient('http://test-api');
    });

    it('get() calls axios.get with correct url and params', async () => {
      mockGet.mockResolvedValue({ data: { success: true } });

      const result = await client.get('/test', { foo: 'bar' });

      expect(result).toEqual({ success: true });
      expect(mockGet).toHaveBeenCalledWith(
        '/test',
        expect.objectContaining({
          params: { foo: 'bar' },
        }),
      );
    });

    it('post() calls axios.post with correct url and body', async () => {
      mockPost.mockResolvedValue({ data: { created: true } });

      const result = await client.post('/test', { body: 'data' });

      expect(result).toEqual({ created: true });
      expect(mockPost).toHaveBeenCalledWith(
        '/test',
        { body: 'data' },
      );
    });
  });

  describe('handleMcpError', () => {
    it('handles 404 AxiosError', () => {
      const error = new AxiosError('Not Found', 'ERR_BAD_REQUEST', undefined, undefined, {
        status: 404,
        statusText: 'Not Found',
        data: {},
        headers: {},
        config: { headers: new AxiosHeaders() },
      });
      expect(handleMcpError(error)).toContain('Resource not found');
    });

    it('handles 429 AxiosError', () => {
      const error = new AxiosError('Too Many Requests', 'ERR_RATE_LIMIT', undefined, undefined, {
        status: 429,
        statusText: 'Too Many Requests',
        data: {},
        headers: {},
        config: { headers: new AxiosHeaders() },
      });
      expect(handleMcpError(error)).toContain('Rate limit exceeded');
    });

    it('handles 500 AxiosError', () => {
      const error = new AxiosError('Internal Server Error', 'ERR_SERVER', undefined, undefined, {
        status: 500,
        statusText: 'Internal Server Error',
        data: {},
        headers: {},
        config: { headers: new AxiosHeaders() },
      });
      expect(handleMcpError(error)).toContain('Internal server error on the backend API');
    });

    it('handles timeout AxiosError', () => {
      const error = new AxiosError('timeout', 'ECONNABORTED');
      expect(handleMcpError(error)).toContain('Request timed out');
    });

    it('handles generic Error', () => {
      const error = new Error('Some standard error');
      expect(handleMcpError(error)).toContain('Some standard error');
    });
  });

  describe('formatMarkdownTable', () => {
    it('formats a markdown table correctly', () => {
      const headers = ['Name', 'Value'];
      const rows = [
        ['Test 1', '10'],
        ['Test 2', '20'],
      ];
      const result = formatMarkdownTable(headers, rows);
      expect(result).toBe('| Name | Value |\n| --- | --- |\n| Test 1 | 10 |\n| Test 2 | 20 |');
    });

    it('returns empty string if no rows or headers', () => {
      expect(formatMarkdownTable([], [['a']])).toBe('');
      expect(formatMarkdownTable(['A'], [])).toBe('');
    });
  });
});
