import { HeartRateRepository } from '../ports/heart-rate.repository';
import { HeartRateReading } from '@monorepo/shared-types';

export class HeartRateService {
  constructor(private readonly repository: HeartRateRepository) {}

  async addReading(
    userId: string,
    data: { value: number; isResting?: boolean },
  ): Promise<HeartRateReading> {
    if (data.value < 30 || data.value > 250) {
      throw new Error('Invalid heart rate value');
    }
    return this.repository.save({ userId, ...data });
  }

  analyzeRestingHR(readings: HeartRateReading[]): number {
    const restingReadings = readings.filter((r) => r.isResting);
    if (restingReadings.length === 0) return 0;
    const sum = restingReadings.reduce((acc, r) => acc + r.value, 0);
    return Math.round(sum / restingReadings.length);
  }

  async getHistory(userId: string): Promise<HeartRateReading[]> {
    return this.repository.findByUserId(userId);
  }

  detectAnomalies(
    readings: HeartRateReading[],
  ): { timestamp: Date; value: number; type: string }[] {
    const anomalies: { timestamp: Date; value: number; type: string }[] = [];

    for (const r of readings) {
      if (r.isResting && r.value > 100) {
        anomalies.push({ timestamp: r.timestamp, value: r.value, type: 'TACHYCARDIA_RESTING' });
      }
      if (r.value < 40) {
        anomalies.push({ timestamp: r.timestamp, value: r.value, type: 'BRADYCARDIA' });
      }
    }

    return anomalies;
  }
}
