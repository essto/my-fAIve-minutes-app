import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DietService } from '../domain/services/diet.service';
import type { DietRepository } from '../domain/ports/diet.repository';
import type { MealEntry } from '@monorepo/shared-types';

const mockDailyTarget = {
  calories: 2000,
  protein: 150, // grams
  carbs: 250, // grams
  fat: 70, // grams
};

describe('DietService (Nutrition Analysis)', () => {
  let dietService: DietService;
  const mockRepository: DietRepository = {
    save: vi.fn(),
    findByUserId: vi.fn(),
  };

  beforeEach(() => {
    vi.resetAllMocks();
    dietService = new DietService(mockRepository);
  });

  describe('analyzeDailyNutrition()', () => {
    it('should return zero totals and full deficits for empty meal history', async () => {
      // Arrange
      (mockRepository.findByUserId as any).mockResolvedValue([]);
      const date = new Date('2023-06-15');

      // Act
      const result = await (dietService as any).analyzeDailyNutrition(
        'user1',
        date,
        mockDailyTarget,
      );

      // Assert
      expect(result).toEqual({
        totalCalories: 0,
        totalProtein: 0,
        totalCarbs: 0,
        totalFat: 0,
        calorieDeficit: 2000,
        proteinDeficit: 150,
        carbsDeficit: 250,
        fatDeficit: 70,
      });
    });

    it('should calculate correct totals and deficits for multiple meals', async () => {
      // Arrange
      const meals: MealEntry[] = [
        {
          id: '1',
          userId: 'user1',
          name: 'Breakfast',
          calories: 600,
          protein: 30,
          carbs: 80,
          fat: 20,
          timestamp: new Date('2023-06-15T08:00:00Z'),
          createdAt: new Date(),
        },
        {
          id: '2',
          userId: 'user1',
          name: 'Lunch',
          calories: 800,
          protein: 40,
          carbs: 90,
          fat: 30,
          timestamp: new Date('2023-06-15T13:00:00Z'),
          createdAt: new Date(),
        },
      ];
      (mockRepository.findByUserId as any).mockResolvedValue(meals);
      const date = new Date('2023-06-15');

      // Act
      const result = await (dietService as any).analyzeDailyNutrition(
        'user1',
        date,
        mockDailyTarget,
      );

      // Assert
      expect(result.totalCalories).toBe(1400);
      expect(result.totalProtein).toBe(70);
      expect(result.calorieDeficit).toBe(600);
      expect(result.proteinDeficit).toBe(80);
    });

    it('should handle negative deficits (overconsumption)', async () => {
      // Arrange
      const meals: MealEntry[] = [
        {
          id: '1',
          userId: 'user1',
          name: 'Big Meal',
          calories: 2500,
          protein: 200,
          carbs: 300,
          fat: 100,
          timestamp: new Date('2023-06-15T12:00:00Z'),
          createdAt: new Date(),
        },
      ];
      (mockRepository.findByUserId as any).mockResolvedValue(meals);
      const date = new Date('2023-06-15');

      // Act
      const result = await (dietService as any).analyzeDailyNutrition(
        'user1',
        date,
        mockDailyTarget,
      );

      // Assert
      expect(result.calorieDeficit).toBe(-500);
      expect(result.proteinDeficit).toBe(-50);
    });

    it('should filter meals by exact date', async () => {
      // Arrange
      const meals: MealEntry[] = [
        {
          id: '1',
          userId: 'user1',
          name: 'Correct Day',
          calories: 500,
          protein: 20,
          carbs: 60,
          fat: 15,
          timestamp: new Date('2023-06-15T10:00:00Z'),
          createdAt: new Date(),
        },
        {
          id: '2',
          userId: 'user1',
          name: 'Wrong Day',
          calories: 1000,
          protein: 50,
          carbs: 100,
          fat: 30,
          timestamp: new Date('2023-06-14T10:00:00Z'),
          createdAt: new Date(),
        },
      ];
      (mockRepository.findByUserId as any).mockResolvedValue(meals);
      const date = new Date('2023-06-15');

      // Act
      const result = await (dietService as any).analyzeDailyNutrition(
        'user1',
        date,
        mockDailyTarget,
      );

      // Assert
      expect(result.totalCalories).toBe(500);
    });
  });
});
