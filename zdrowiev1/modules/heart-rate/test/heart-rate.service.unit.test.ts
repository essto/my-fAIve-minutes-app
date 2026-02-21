import { describe, expect, it, vi, beforeEach } from 'vitest';
import { HeartRateService } from '../domain/services/heart-rate.service';
import type { HeartRateRepository } from '../domain/ports/heart-rate.repository';
import type { HeartRateReading } from '@monorepo/shared-types';

describe('HeartRateService (Domain Logic)', () => {
  let service: HeartRateService;
  const mockRepository: HeartRateRepository = {
    save: vi.fn(),
    findByUserId: vi.fn(),
  };

  beforeEach(() => {
    vi.resetAllMocks();
    service = new HeartRateService(mockRepository);
  });

  describe('analyzeRestingHR', () => {
    it('returns 0 for no resting readings', () => {
      const readings: HeartRateReading[] = [
        {
          id: '1',
          userId: 'u1',
          value: 150,
          isResting: false,
          timestamp: new Date(),
          createdAt: new Date(),
        },
      ] as any;
      expect(service.analyzeRestingHR(readings)).toBe(0);
    });

    it('calculates average resting heart rate correctly', () => {
      const readings: HeartRateReading[] = [
        { value: 60, isResting: true },
        { value: 70, isResting: true },
        { value: 80, isResting: false }, // Should be ignored
      ] as any;
      expect(service.analyzeRestingHR(readings)).toBe(65);
    });
  });

  describe('detectAnomalies', () => {
    it('detects TACHYCARDIA_RESTING when resting HR > 100', () => {
      const readings: HeartRateReading[] = [
        { value: 110, isResting: true, timestamp: new Date('2023-01-01T10:00:00Z') },
      ] as any;
      const anomalies = service.detectAnomalies(readings);
      expect(anomalies).toHaveLength(1);
      expect(anomalies[0].type).toBe('TACHYCARDIA_RESTING');
    });

    it('detects BRADYCARDIA when HR < 40', () => {
      const readings: HeartRateReading[] = [
        { value: 35, isResting: false, timestamp: new Date('2023-01-01T11:00:00Z') },
      ] as any;
      const anomalies = service.detectAnomalies(readings);
      expect(anomalies).toHaveLength(1);
      expect(anomalies[0].type).toBe('BRADYCARDIA');
    });

    it('returns empty array when no anomalies detected', () => {
      const readings: HeartRateReading[] = [
        { value: 70, isResting: true, timestamp: new Date() },
        { value: 140, isResting: false, timestamp: new Date() },
      ] as any;
      expect(service.detectAnomalies(readings)).toHaveLength(0);
    });
  });
});
