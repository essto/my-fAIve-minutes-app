export type LogEntry = {
  id: string;
  timestamp: string;
  level: 'ERROR' | 'WARN' | 'INFO';
  message: string;
  stackTrace?: string;
};

export class LoggerService {
  private logs: LogEntry[] = [
    {
      id: 'log-1',
      timestamp: new Date().toISOString(),
      level: 'ERROR',
      message: 'Database connection timeout in Weight module',
      stackTrace: 'at connect (db.ts:25)\n  at fetch (weight.ts:14)',
    },
    {
      id: 'log-2',
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      level: 'WARN',
      message: 'High memory usage in SymptomChecker',
    },
  ];

  async getRecentErrors(limit: number, level?: string): Promise<LogEntry[]> {
    let filtered = this.logs;
    if (level) {
      filtered = filtered.filter((l) => l.level === level);
    }
    return filtered.slice(0, limit);
  }

  async getErrorPatterns(hours: number): Promise<{ pattern: string; count: number }[]> {
    return [
      { pattern: 'Database connection timeout', count: 12 },
      { pattern: 'Null pointer exception in Diet api', count: 4 },
    ];
  }

  async suggestFix(pattern: string): Promise<string> {
    if (pattern.includes('Database')) {
      return 'Increase connection pool size in drizzle.config.ts or verify PostgreSQL max_connections.';
    }
    return 'Analiza wymga więcej logów. Spróbuj dodać więcej kontekstu do wywołań w module zgłaszającym błąd.';
  }
}
