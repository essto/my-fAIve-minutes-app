import { describe, expect, it, vi, beforeEach } from 'vitest';
import { SleepService } from '../domain/services/sleep.service';
import type { SleepRepository } from '../domain/ports/sleep.repository';

describe('SleepService (Domain Logic)', () => {
  let service: SleepService;
  const mockRepository: SleepRepository = {
    save: vi.fn(),
    findByUserId: vi.fn(),
  };

  beforeEach(() => {
    vi.resetAllMocks();
    service = new SleepService(mockRepository);
  });

  describe('calculateSleepScore', () => {
    it('throws for invalid time range (end before start)', () => {
      const start = new Date('2023-01-01T22:00:00');
      const end = new Date('2023-01-01T21:00:00');
      expect(() => (service as any).calculateSleepScore(start, end, 0.8)).toThrowError(
        'End time must be after start',
      );
    });

    it('calculates perfect score (100) for 8h and 100% quality', () => {
      const start = new Date('2023-01-01T22:00:00');
      const end = new Date('2023-01-02T06:00:00'); // 8 hours
      expect((service as any).calculateSleepScore(start, end, 1.0)).toBe(100);
    });

    it('calculates score for 8h and 75% quality', () => {
      const start = new Date('2023-01-01T22:00:00');
      const end = new Date('2023-01-02T06:00:00');
      const score = (service as any).calculateSleepScore(start, end, 0.75);
      expect(score).toBe(75);
    });

    it('penalizes short sleep duration (e.g. 4 hours)', () => {
      const start = new Date('2023-01-01T02:00:00');
      const end = new Date('2023-01-01T06:00:00'); // 4 hours
      const score = (service as any).calculateSleepScore(start, end, 1.0);
      // 4h is 50% of target 8h. 100 quality * 0.5 duration factor = 50
      expect(score).toBe(50);
    });
  });
});
