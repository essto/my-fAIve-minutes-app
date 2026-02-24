import { WeightRepository } from '../ports/weight.repository';
import { WeightReading } from '@monorepo/shared-types';

export class WeightService {
  constructor(private readonly repository: WeightRepository) {}

  async addReading(
    userId: string,
    data: { value: number; unit?: 'kg' | 'lbs'; source?: string },
  ): Promise<WeightReading> {
    if (data.value <= 0 || data.value > 500) {
      throw new Error('Invalid weight value');
    }
    return this.repository.save({ userId, ...data });
  }

  calculateBMI(weightKg: number, heightCm: number): number {
    if (heightCm <= 0) throw new Error('Invalid height');
    const heightM = heightCm / 100;
    const bmi = weightKg / (heightM * heightM);
    return Math.round(bmi * 10) / 10;
  }

  async getHistory(userId: string): Promise<WeightReading[]> {
    return this.repository.findByUserId(userId);
  }

  analyzeWeightTrend(readings: { value: number; timestamp: Date }[]): { slope: number } {
    if (readings.length < 2) {
      throw new Error('At least 2 records required');
    }

    // Simple linear regression to find the slope
    // X = time in days, Y = weight
    const sorted = [...readings].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
    const firstTime = sorted[0].timestamp.getTime();

    const n = sorted.length;
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumX2 = 0;

    for (const r of sorted) {
      const x = (r.timestamp.getTime() - firstTime) / (1000 * 60 * 60 * 24); // days
      const y = r.value;
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumX2 += x * x;
    }

    const divisor = n * sumX2 - sumX * sumX;
    const slope = divisor === 0 ? 0 : (n * sumXY - sumX * sumY) / divisor;

    return { slope };
  }
}
