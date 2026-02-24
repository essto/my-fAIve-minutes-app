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
        value: data.value!,
        isResting: data.isResting,
        timestamp: data.timestamp,
      })
      .returning();

    return {
      id: inserted.id,
      userId: inserted.userId,
      value: inserted.value,
      isResting: inserted.isResting ?? false,
      timestamp: inserted.timestamp ?? new Date(),
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
      value: row.value,
      isResting: row.isResting ?? false,
      timestamp: row.timestamp ?? new Date(),
    }));
  }
}
