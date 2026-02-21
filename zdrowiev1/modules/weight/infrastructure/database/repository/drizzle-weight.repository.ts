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
      .values(data as any)
      .returning();
    return inserted as unknown as WeightReading;
  }

  async findByUserId(userId: string): Promise<WeightReading[]> {
    const results = await this.db
      .select()
      .from(weightReadings)
      .where(eq(weightReadings.userId, userId));
    return results as unknown as WeightReading[];
  }
}
