import { MealRepository } from '../../domain/ports/diet.repository';
import { Meal } from '../../domain/entities/meal.entity';
import { NutritionCalculator } from '../../domain/services/nutrition-calculator';

export class MealService {
  private readonly calculator = new NutritionCalculator();

  constructor(private readonly repository: MealRepository) {}

  async logMeal(userId: string, name: string, products: Meal['products']): Promise<Meal> {
    const meal: Meal = {
      id: crypto.randomUUID(),
      userId,
      name,
      consumedAt: new Date(),
      products,
    };

    return this.repository.save(meal);
  }

  async getDailySummary(userId: string, date: Date = new Date()) {
    const startOfDay = new Date(date);
    startOfDay.setHours(0, 0, 0, 0);

    const meals = await this.repository.findByUserId(userId, startOfDay);
    const total = this.calculator.calculateDailyTotal(meals);

    // In a real app, target would come from user profile/service
    const mockTarget = { calories: 2000, protein: 150, carbs: 200, fat: 70 };
    const status = this.calculator.calculateDailyStatus(total, mockTarget);

    return {
      date: startOfDay.toISOString().split('T')[0],
      total,
      target: mockTarget,
      ...status,
    };
  }

  async getMealById(id: string): Promise<Meal | null> {
    return this.repository.findById(id);
  }
}
