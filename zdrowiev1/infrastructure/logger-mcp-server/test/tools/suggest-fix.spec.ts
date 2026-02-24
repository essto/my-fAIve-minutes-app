import { describe, it, expect } from 'vitest';
import { handleSuggestFix } from '../../src/tools/suggest-fix.tool.js';
import { LoggerService } from '../../src/logger.service.js';

describe('logger_suggest_fix tool', () => {
  it('returns a suggestion based on pattern string', async () => {
    const service = new LoggerService();
    const result = await handleSuggestFix(
      { error_pattern: 'Database connection timeout' },
      service,
    );

    expect(result.content[0].text).toContain('drizzle.config.ts');
  });

  it('returns generic fallback if not found', async () => {
    const service = new LoggerService();
    const result = await handleSuggestFix({ error_pattern: 'unknown obscure error' }, service);

    expect(result.content[0].text).toContain('więcej logów');
  });
});
