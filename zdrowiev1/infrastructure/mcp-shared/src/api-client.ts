import axios, { AxiosError, AxiosInstance } from 'axios';

export const CHARACTER_LIMIT = 25000;
export const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:3001';

export class McpApiClient {
  private client: AxiosInstance;

  constructor(baseUrl: string = API_BASE_URL) {
    this.client = axios.create({
      baseURL: baseUrl,
      timeout: 30000,
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
    });
  }

  async get<T>(path: string, params?: Record<string, any>): Promise<T> {
    const response = await this.client.get<T>(path, { params });
    return response.data;
  }

  async post<T>(path: string, data?: any): Promise<T> {
    const response = await this.client.post<T>(path, data);
    return response.data;
  }
}

export function handleMcpError(error: unknown): string {
  if (error instanceof AxiosError) {
    if (error.response) {
      switch (error.response.status) {
        case 404:
          return 'Error: Resource not found. Please check the provided IDs.';
        case 403:
          return "Error: Permission denied. You don't have access to this resource.";
        case 429:
          return 'Error: Rate limit exceeded. Please wait before making more requests.';
        case 500:
          return 'Error: Internal server error on the backend API.';
        default:
          return `Error: API request failed with status ${error.response.status}. ${error.response.data?.message || ''}`;
      }
    } else if (error.code === 'ECONNABORTED') {
      return 'Error: Request timed out. Please try again.';
    }
  }
  return `Error: Unexpected error occurred: ${error instanceof Error ? error.message : String(error)}`;
}

export function formatMarkdownTable(headers: string[], rows: any[][]): string {
  if (!headers.length || !rows.length) return '';

  const headerRow = `| ${headers.join(' | ')} |`;
  const separatorRow = `| ${headers.map(() => '---').join(' | ')} |`;
  const bodyRows = rows.map((row) => `| ${row.join(' | ')} |`).join('\n');

  return `${headerRow}\n${separatorRow}\n${bodyRows}`;
}
