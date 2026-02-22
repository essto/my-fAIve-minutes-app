import { Meal } from '../entities/meal.entity';

export interface MealRepository {
  save(meal: Meal): Promise<Meal>;
  findById(id: string): Promise<Meal | null>;
  findByUserId(userId: string, date?: Date): Promise<Meal[]>;
}
