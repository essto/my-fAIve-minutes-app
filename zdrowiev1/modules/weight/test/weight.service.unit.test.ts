import { describe, expect, it, vi, beforeEach } from 'vitest';
import { WeightService } from '../domain/services/weight.service';
import type { WeightRepository } from '../domain/ports/weight.repository';

describe('WeightService (Domain Logic)', () => {
  let service: WeightService;
  const mockRepository: WeightRepository = {
    save: vi.fn(),
    findByUserId: vi.fn(),
  };

  beforeEach(() => {
    vi.resetAllMocks();
    service = new WeightService(mockRepository);
  });

  describe('calculateBMI', () => {
    it('throws error for zero height', () => {
      expect(() => service.calculateBMI(70, 0)).toThrowError('Invalid height');
    });

    it('returns correct BMI for valid inputs', () => {
      expect(service.calculateBMI(70, 175)).toBeCloseTo(22.9, 1);
    });
  });

  describe('analyzeWeightTrend', () => {
    it('throws for insufficient data points (less than 2)', () => {
      expect(() =>
        (service as any).analyzeWeightTrend([
          {
            value: 70,
            timestamp: new Date(),
          },
        ]),
      ).toThrowError('At least 2 records required');
    });

    it('calculates downward trend slope correctly', () => {
      const records = [
        { value: 80, timestamp: new Date('2023-01-01') },
        { value: 75, timestamp: new Date('2023-01-08') },
        { value: 70, timestamp: new Date('2023-01-15') },
      ];
      const trend = (service as any).analyzeWeightTrend(records);
      expect(trend.slope).toBeLessThan(0); // Negative slope
      // Δweight = 70 - 80 = -10, Δdays = 14 => slope = -10/14 ≈ -0.71
      expect(trend.slope).toBeCloseTo(-0.71, 2);
    });

    it('calculates upward trend slope correctly', () => {
      const records = [
        { value: 70, timestamp: new Date('2023-01-01') },
        { value: 72, timestamp: new Date('2023-01-08') },
        { value: 74, timestamp: new Date('2023-01-15') },
      ];
      const trend = (service as any).analyzeWeightTrend(records);
      expect(trend.slope).toBeGreaterThan(0);
      expect(trend.slope).toBeCloseTo(4 / 14, 2);
    });
  });
  describe('getHealthSummary', () => {
    it('returns current weight, change30d, and bmi correctly based on history', async () => {
      const now = new Date();
      const d31ago = new Date(now);
      d31ago.setDate(now.getDate() - 31);

      (mockRepository.findByUserId as any).mockResolvedValue([
        { value: 80.0, bmi: 25.0, fatPercent: 20.0, timestamp: now },
        { value: 82.0, bmi: 25.5, fatPercent: 20.5, timestamp: d31ago },
      ]);

      const summary = await service.getHealthSummary('u1');

      expect(summary.current).toBe(80.0);
      expect(summary.change30d).toBeCloseTo(-2.0);
      expect(summary.bmi).toBe(25.0);
    });

    it('handles empty history safely', async () => {
      (mockRepository.findByUserId as any).mockResolvedValue([]);
      const summary = await service.getHealthSummary('u1');
      expect(summary).toEqual({ current: 0, change30d: 0, bmi: 0 });
    });
  });
});
