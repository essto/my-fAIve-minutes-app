import { SleepRepository } from '../ports/sleep.repository';
import { SleepRecord } from '@monorepo/shared-types';

export class SleepService {
  constructor(private readonly repository: SleepRepository) {}

  async addRecord(
    userId: string,
    data: { startTime: Date; endTime: Date; quality?: number },
  ): Promise<SleepRecord> {
    if (data.endTime <= data.startTime) {
      throw new Error('End time must be after start time');
    }
    return this.repository.save({ userId, ...data });
  }

  calculateEfficiency(records: SleepRecord[]): number {
    if (records.length === 0) return 0;
    // Simple logic: average quality or duration vs target
    const totalQuality = records.reduce((acc, r) => acc + (r.quality || 5), 0);
    return Math.round(totalQuality / records.length);
  }

  async getHistory(userId: string): Promise<SleepRecord[]> {
    return this.repository.findByUserId(userId);
  }

  calculateSleepScore(startTime: Date, endTime: Date, qualityFactor: number): number {
    if (endTime <= startTime) {
      throw new Error('End time must be after start');
    }

    const durationHours = (endTime.getTime() - startTime.getTime()) / (1000 * 60 * 60);
    const targetHours = 8;

    const durationFactor = Math.min(durationHours / targetHours, 1.0);
    const score = Math.round(100 * durationFactor * qualityFactor);

    return score;
  }
}
