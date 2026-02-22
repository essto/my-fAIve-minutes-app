import { ActivityEntry } from '@monorepo/shared-types';

export interface IActivityRepository {
  save(entry: Partial<ActivityEntry>): Promise<ActivityEntry>;
  findByUserIdAndDate(userId: string, date: string): Promise<ActivityEntry[]>;
  getDailySummary(userId: string, date: string): Promise<ActivityEntry | null>;
}
