import { describe, it, expect } from 'vitest';
import { NutritionCalculator } from '../services/nutrition-calculator';
import { Meal, MealProduct } from '../entities/meal.entity';

describe('NutritionCalculator', () => {
  const calculator = new NutritionCalculator();

  const mockProduct: MealProduct = {
    name: 'Twaróg',
    quantity: 100,
    calories: 100,
    protein: 18,
    carbs: 3,
    fat: 2,
  };

  const mockMeal: Meal = {
    id: 'm1',
    userId: 'u1',
    name: 'Śniadanie',
    consumedAt: new Date(),
    products: [mockProduct, mockProduct], // 200 cal, 36 protein, 6 carbs, 4 fat
  };

  it('should correctly sum nutrition values for a single meal', () => {
    const summary = calculator.calculateMealNutrition(mockMeal);
    expect(summary.calories).toBe(200);
    expect(summary.protein).toBe(36);
    expect(summary.carbs).toBe(6);
    expect(summary.fat).toBe(4);
  });

  it('should correctly sum nutrition values for multiple meals', () => {
    const meals = [mockMeal, mockMeal];
    const summary = calculator.calculateDailyTotal(meals);
    expect(summary.calories).toBe(400);
    expect(summary.protein).toBe(72);
    expect(summary.carbs).toBe(12);
    expect(summary.fat).toBe(8);
  });

  it('should detect deficit when calories are significantly below target', () => {
    const consumed = { calories: 1500, protein: 100, carbs: 150, fat: 50 };
    const target = { calories: 2000, protein: 150, carbs: 200, fat: 70 };

    const status = calculator.calculateDailyStatus(consumed, target);
    expect(status.isDeficit).toBe(true);
    expect(status.isSurplus).toBe(false);
  });

  it('should detect surplus when calories are significantly above target', () => {
    const consumed = { calories: 2500, protein: 100, carbs: 150, fat: 50 };
    const target = { calories: 2000, protein: 150, carbs: 200, fat: 70 };

    const status = calculator.calculateDailyStatus(consumed, target);
    expect(status.isDeficit).toBe(false);
    expect(status.isSurplus).toBe(true);
  });

  it('should be balanced when calories are close to target', () => {
    const consumed = { calories: 2050, protein: 100, carbs: 150, fat: 50 };
    const target = { calories: 2000, protein: 150, carbs: 200, fat: 70 };

    const status = calculator.calculateDailyStatus(consumed, target);
    expect(status.isDeficit).toBe(false);
    expect(status.isSurplus).toBe(false);
  });
});
