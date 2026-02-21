import { db } from '../../../../shared/database/src/index';
import { mealEntries } from '../../schemas/diet.schema';
import { DietRepository } from '../../../domain/ports/diet.repository';
import { MealEntry } from '@monorepo/shared-types';
import { eq } from 'drizzle-orm';

export class DrizzleDietRepository implements DietRepository {
  async save(data: Partial<MealEntry>): Promise<MealEntry> {
    const [inserted] = await db
      .insert(mealEntries)
      .values(data as any)
      .returning();
    return inserted as unknown as MealEntry;
  }

  async findByUserId(userId: string): Promise<MealEntry[]> {
    const results = await db.select().from(mealEntries).where(eq(mealEntries.userId, userId));
    return results as unknown as MealEntry[];
  }
}
