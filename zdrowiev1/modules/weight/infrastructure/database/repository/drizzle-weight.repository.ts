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
        value: data.value!,
        unit: data.unit ?? 'kg',
        bmi: data.bmi,
        fatPercent: data.fatPercent,
        fatKg: data.fatKg,
        muscleMassKg: data.muscleMassKg,
        musclePercent: data.musclePercent,
        waterPercent: data.waterPercent,
        bmrKcal: data.bmrKcal,
        boneMassKg: data.boneMassKg,
        proteinPercent: data.proteinPercent,
        leanMassKg: data.leanMassKg,
        metabolicAge: data.metabolicAge,
        timestamp: data.timestamp,
      })
      .returning();

    return {
      id: inserted.id,
      userId: inserted.userId,
      value: inserted.value,
      unit: inserted.unit as 'kg' | 'lbs',
      bmi: inserted.bmi ?? undefined,
      fatPercent: inserted.fatPercent ?? undefined,
      fatKg: inserted.fatKg ?? undefined,
      muscleMassKg: inserted.muscleMassKg ?? undefined,
      musclePercent: inserted.musclePercent ?? undefined,
      waterPercent: inserted.waterPercent ?? undefined,
      bmrKcal: inserted.bmrKcal ?? undefined,
      boneMassKg: inserted.boneMassKg ?? undefined,
      proteinPercent: inserted.proteinPercent ?? undefined,
      leanMassKg: inserted.leanMassKg ?? undefined,
      metabolicAge: inserted.metabolicAge ?? undefined,
      timestamp: inserted.timestamp ?? new Date(),
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
      value: row.value,
      unit: row.unit as 'kg' | 'lbs',
      bmi: row.bmi ?? undefined,
      fatPercent: row.fatPercent ?? undefined,
      fatKg: row.fatKg ?? undefined,
      muscleMassKg: row.muscleMassKg ?? undefined,
      musclePercent: row.musclePercent ?? undefined,
      waterPercent: row.waterPercent ?? undefined,
      bmrKcal: row.bmrKcal ?? undefined,
      boneMassKg: row.boneMassKg ?? undefined,
      proteinPercent: row.proteinPercent ?? undefined,
      leanMassKg: row.leanMassKg ?? undefined,
      metabolicAge: row.metabolicAge ?? undefined,
      timestamp: row.timestamp ?? new Date(),
    }));
  }
}
