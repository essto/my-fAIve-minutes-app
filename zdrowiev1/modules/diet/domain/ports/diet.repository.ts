import { MealEntry } from '@monorepo/shared-types';

export interface DietRepository {
  save(entry: Partial<MealEntry>): Promise<MealEntry>;
  findByUserId(userId: string): Promise<MealEntry[]>;
}
