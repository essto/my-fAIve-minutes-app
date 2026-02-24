import { db } from '@monorepo/database';
import { heartRateReadings } from '../../schemas/heart-rate.schema';
import { HeartRateRepository } from '../../../domain/ports/heart-rate.repository';
import { HeartRateReading } from '@monorepo/shared-types';
import { eq } from 'drizzle-orm';

export class DrizzleHeartRateRepository implements HeartRateRepository {
  async save(data: Partial<HeartRateReading>): Promise<HeartRateReading> {
    const [inserted] = await db
      .insert(heartRateReadings)
      .values({
        userId: data.userId!,
        bpm: data.bpm!,
        isResting: data.isResting,
        measuredAt: data.measuredAt,
      })
      .returning();

    return {
      id: inserted.id,
      userId: inserted.userId,
      bpm: inserted.bpm,
      isResting: inserted.isResting ?? false,
      measuredAt: inserted.measuredAt ?? new Date(),
    };
  }

  async findByUserId(userId: string): Promise<HeartRateReading[]> {
    const results = await db
      .select()
      .from(heartRateReadings)
      .where(eq(heartRateReadings.userId, userId));

    return results.map((row) => ({
      id: row.id,
      userId: row.userId,
      bpm: row.bpm,
      isResting: row.isResting ?? false,
      measuredAt: row.measuredAt ?? new Date(),
    }));
  }
}
