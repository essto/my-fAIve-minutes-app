import { DietRepository } from '../ports/diet.repository';
import { MealEntry } from '@monorepo/shared-types';

export class DietService {
  constructor(private readonly repository: DietRepository) {}

  async addEntry(
    userId: string,
    data: Omit<MealEntry, 'id' | 'userId' | 'timestamp'>,
  ): Promise<MealEntry> {
    if (data.calories < 0) {
      throw new Error('Calories cannot be negative');
    }
    return this.repository.save({ userId, ...data });
  }

  calculateDailyCalories(entries: MealEntry[], date: Date): number {
    const startOfDay = new Date(date);
    startOfDay.setHours(0, 0, 0, 0);
    const endOfDay = new Date(date);
    endOfDay.setHours(23, 59, 59, 999);

    return entries
      .filter((e) => e.timestamp >= startOfDay && e.timestamp <= endOfDay)
      .reduce((acc, e) => acc + e.calories, 0);
  }

  async getHistory(userId: string): Promise<MealEntry[]> {
    return this.repository.findByUserId(userId);
  }

  async analyzeDailyNutrition(
    userId: string,
    date: Date,
    target: { calories: number; protein: number; carbs: number; fat: number },
  ) {
    const startOfDay = new Date(date);
    startOfDay.setUTCHours(0, 0, 0, 0);
    const endOfDay = new Date(date);
    endOfDay.setUTCHours(23, 59, 59, 999);

    const meals = await this.repository.findByUserId(userId);

    const dayMeals = meals.filter((e) => {
      const timestamp = new Date(e.timestamp);
      return timestamp >= startOfDay && timestamp <= endOfDay;
    });

    const totals = dayMeals.reduce(
      (acc, e) => ({
        totalCalories: acc.totalCalories + (e.calories || 0),
        totalProtein: acc.totalProtein + (e.protein || 0),
        totalCarbs: acc.totalCarbs + (e.carbs || 0),
        totalFat: acc.totalFat + (e.fat || 0),
      }),
      { totalCalories: 0, totalProtein: 0, totalCarbs: 0, totalFat: 0 },
    );

    return {
      ...totals,
      calorieDeficit: target.calories - totals.totalCalories,
      proteinDeficit: target.protein - totals.totalProtein,
      carbsDeficit: target.carbs - totals.totalCarbs,
      fatDeficit: target.fat - totals.totalFat,
    };
  }
}
