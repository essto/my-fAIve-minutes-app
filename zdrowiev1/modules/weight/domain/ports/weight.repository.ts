import { WeightReading } from '@monorepo/shared-types';

export interface WeightRepository {
  save(reading: Partial<WeightReading>): Promise<WeightReading>;
  findByUserId(userId: string): Promise<WeightReading[]>;
}
