import { db } from '@monorepo/database';
import { sleepRecords } from '../../schemas/sleep.schema';
import { SleepRepository } from '../../../domain/ports/sleep.repository';
import { SleepRecord } from '@monorepo/shared-types';
import { eq } from 'drizzle-orm';

export class DrizzleSleepRepository implements SleepRepository {
  async save(data: Partial<SleepRecord>): Promise<SleepRecord> {
    const [inserted] = await db
      .insert(sleepRecords)
      .values(data as any)
      .returning();
    return inserted as unknown as SleepRecord;
  }

  async findByUserId(userId: string): Promise<SleepRecord[]> {
    const results = await db.select().from(sleepRecords).where(eq(sleepRecords.userId, userId));
    return results as unknown as SleepRecord[];
  }
}
