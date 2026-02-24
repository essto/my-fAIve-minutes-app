import { describe, it, expect } from 'vitest';
import { handleGetErrorPatterns } from '../../src/tools/get-error-patterns.tool.js';
import { LoggerService } from '../../src/logger.service.js';

describe('logger_get_error_patterns tool', () => {
  it('returns grouped error patterns', async () => {
    const service = new LoggerService();
    const result = await handleGetErrorPatterns({ timeframe_hours: 24 }, service);

    expect(result.content[0].text).toContain('Database connection timeout');
    expect((result.structuredContent as any).length).toBeGreaterThan(0);
  });
});
