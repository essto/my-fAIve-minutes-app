import { describe, it, expect } from 'vitest';
import { handleGetRecentErrors } from '../../src/tools/get-recent-errors.tool.js';
import { LoggerService } from '../../src/logger.service.js';

describe('logger_get_recent_errors tool', () => {
  it('returns exactly limit number of errors when requested', async () => {
    const service = new LoggerService();
    // Assuming our fake service has 2 logs total
    const result = await handleGetRecentErrors({ limit: 1 }, service);
    expect(result.content[0].text).toContain('Znaleziono 1');
    expect((result.structuredContent as any).length).toBe(1);
  });

  it('filters by level', async () => {
    const service = new LoggerService();
    const result = await handleGetRecentErrors({ limit: 10, level: 'ERROR' }, service);
    expect((result.structuredContent as any[])[0].level).toBe('ERROR');
  });
});
