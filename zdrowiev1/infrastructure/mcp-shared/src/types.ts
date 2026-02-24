export type McpToolResult = {
  content: { type: 'text'; text: string }[];
  structuredContent?: unknown;
};

export type PaginatedResponse<T> = {
  total: number;
  count: number;
  offset: number;
  items: T[];
  has_more: boolean;
  next_offset?: number;
};
