import { Injectable, Inject } from '@nestjs/common';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { eq, and } from 'drizzle-orm';
import { IActivityRepository } from '../../domain/ports/activity.repository';
import { activityEntries } from '../schemas/activity.schema';
import { ActivityEntry } from '@monorepo/shared-types';
import { DATABASE_CONNECTION } from '@monorepo/database';

@Injectable()
export class ActivityDrizzleRepository implements IActivityRepository {
  constructor(
    @Inject(DATABASE_CONNECTION)
    private readonly db: NodePgDatabase,
  ) {}

  async save(entry: Partial<ActivityEntry>): Promise<ActivityEntry> {
    const [result] = await this.db
      .insert(activityEntries)
      .values({
        userId: entry.userId!,
        date: entry.date!,
        steps: entry.steps || 0,
        caloriesBurned: entry.caloriesBurned || 0,
        activityType: entry.activityType,
        durationMinutes: entry.durationMinutes,
      })
      .returning();

    return result as ActivityEntry;
  }

  async findByUserIdAndDate(userId: string, date: string): Promise<ActivityEntry[]> {
    const results = await this.db
      .select()
      .from(activityEntries)
      .where(and(eq(activityEntries.userId, userId), eq(activityEntries.date, date)));

    return results as ActivityEntry[];
  }

  async getDailySummary(userId: string, date: string): Promise<ActivityEntry | null> {
    const results = await this.db
      .select()
      .from(activityEntries)
      .where(and(eq(activityEntries.userId, userId), eq(activityEntries.date, date)));

    if (results.length === 0) return null;

    // Aggregate summary
    const summary = results.reduce(
      (acc, curr) => ({
        ...acc,
        steps: acc.steps + curr.steps,
        caloriesBurned: acc.caloriesBurned + curr.caloriesBurned,
      }),
      {
        id: 'daily-summary',
        userId,
        date,
        steps: 0,
        caloriesBurned: 0,
        createdAt: new Date(),
      } as ActivityEntry,
    );

    return summary;
  }
}
