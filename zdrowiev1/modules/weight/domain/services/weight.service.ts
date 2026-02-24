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

  async getHealthSummary(userId: string): Promise<{
    current: number;
    change30d: number;
    bmi: number | null;
  }> {
    const all = await this.repository.findByUserId(userId);
    if (!all || all.length === 0) {
      return { current: 0, change30d: 0, bmi: 0 };
    }
    const sorted = [...all].sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
    );
    const latest = sorted[0];
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
    const baseline = sorted.find((r) => new Date(r.timestamp) <= thirtyDaysAgo);
    const change30d = baseline ? +(latest.value - baseline.value).toFixed(1) : 0;
    return {
      current: latest.value,
      change30d,
      bmi: latest.bmi ?? null,
    };
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
