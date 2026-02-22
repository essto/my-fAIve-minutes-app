import { db } from '@monorepo/database';
import { heartRateReadings } from '../../schemas/heart-rate.schema';
import { HeartRateRepository } from '../../../domain/ports/heart-rate.repository';
import { HeartRateReading } from '@monorepo/shared-types';
import { eq } from 'drizzle-orm';

export class DrizzleHeartRateRepository implements HeartRateRepository {
  async save(data: Partial<HeartRateReading>): Promise<HeartRateReading> {
    const [inserted] = await db
      .insert(heartRateReadings)
      .values(data as any)
      .returning();
    return inserted as unknown as HeartRateReading;
  }

  async findByUserId(userId: string): Promise<HeartRateReading[]> {
    const results = await db
      .select()
      .from(heartRateReadings)
      .where(eq(heartRateReadings.userId, userId));
    return results as unknown as HeartRateReading[];
  }
}
