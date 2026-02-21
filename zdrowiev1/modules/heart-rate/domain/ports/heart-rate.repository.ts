import { HeartRateReading } from '@monorepo/shared-types';

export interface HeartRateRepository {
  save(reading: Partial<HeartRateReading>): Promise<HeartRateReading>;
  findByUserId(userId: string): Promise<HeartRateReading[]>;
}
