import { SleepRecord } from '@monorepo/shared-types';

export interface SleepRepository {
  save(record: Partial<SleepRecord>): Promise<SleepRecord>;
  findByUserId(userId: string): Promise<SleepRecord[]>;
}
