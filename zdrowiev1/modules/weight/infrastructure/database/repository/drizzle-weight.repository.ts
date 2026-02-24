import { WeightRepository } from '../../../domain/ports/weight.repository';
import { WeightReading } from '@monorepo/shared-types';
import { eq } from 'drizzle-orm';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { weightReadings } from '../../schemas/weight.schema';

export class DrizzleWeightRepository implements WeightRepository {
  constructor(private readonly db: NodePgDatabase<any>) {}

  async save(data: Partial<WeightReading>): Promise<WeightReading> {
    const [inserted] = await this.db
      .insert(weightReadings)
      .values({
        userId: data.userId!,
        weight: data.weight!,
        unit: data.unit ?? 'kg',
        measuredAt: data.measuredAt,
      })
      .returning();

    return {
      id: inserted.id,
      userId: inserted.userId,
      weight: inserted.weight,
      unit: inserted.unit as 'kg' | 'lbs',
      measuredAt: inserted.measuredAt ?? new Date(),
    };
  }

  async findByUserId(userId: string): Promise<WeightReading[]> {
    const results = await this.db
      .select()
      .from(weightReadings)
      .where(eq(weightReadings.userId, userId));

    return results.map((row) => ({
      id: row.id,
      userId: row.userId,
      weight: row.weight,
      unit: row.unit as 'kg' | 'lbs',
      measuredAt: row.measuredAt ?? new Date(),
    }));
  }
}
