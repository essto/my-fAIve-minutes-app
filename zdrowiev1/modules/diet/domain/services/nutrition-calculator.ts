import { Meal, NutritionValue } from '../entities/meal.entity';

export class NutritionCalculator {
  calculateMealNutrition(meal: Meal): NutritionValue {
    return meal.products.reduce(
      (acc, product) => ({
        calories: acc.calories + product.calories,
        protein: acc.protein + product.protein,
        carbs: acc.carbs + product.carbs,
        fat: acc.fat + product.fat,
      }),
      { calories: 0, protein: 0, carbs: 0, fat: 0 },
    );
  }

  calculateDailyTotal(meals: Meal[]): NutritionValue {
    return meals.reduce(
      (acc, meal) => {
        const mealNutrition = this.calculateMealNutrition(meal);
        return {
          calories: acc.calories + mealNutrition.calories,
          protein: acc.protein + mealNutrition.protein,
          carbs: acc.carbs + mealNutrition.carbs,
          fat: acc.fat + mealNutrition.fat,
        };
      },
      { calories: 0, protein: 0, carbs: 0, fat: 0 },
    );
  }

  calculateDailyStatus(consumed: NutritionValue, target: NutritionValue) {
    const diff = consumed.calories - target.calories;
    return {
      isDeficit: diff < -200,
      isSurplus: diff > 200,
      balance: {
        calories: diff,
        protein: consumed.protein - target.protein,
        carbs: consumed.carbs - target.carbs,
        fat: consumed.fat - target.fat,
      },
    };
  }
}
