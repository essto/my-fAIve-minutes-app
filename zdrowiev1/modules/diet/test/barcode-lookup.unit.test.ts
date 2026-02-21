import { beforeEach, describe, expect, it, vi } from 'vitest';
import { BarcodeLookupService } from '../domain/services/barcode-lookup.service';
import type { FoodDatabasePort } from '../domain/ports/food-database.port';
import type { CachePort } from '../domain/ports/cache.port';
import type { FoodItem } from '../domain/entities/food-item.entity';

describe('BarcodeLookupService', () => {
  let lookupService: BarcodeLookupService;
  const mockFoodDatabase: FoodDatabasePort = {
    fetchFoodItem: vi.fn(),
  };
  const mockCache: CachePort = {
    get: vi.fn(),
    set: vi.fn(),
  };

  beforeEach(() => {
    vi.resetAllMocks();
    lookupService = new BarcodeLookupService(mockFoodDatabase, mockCache);
  });

  describe('lookupBarcode()', () => {
    const testBarcode = '123456789';
    const cachedFoodItem: FoodItem = {
      barcode: testBarcode,
      name: 'Cached Food',
      calories: 100,
      protein: 10,
      carbs: 20,
      fat: 2,
    };
    const freshFoodItem: FoodItem = {
      barcode: testBarcode,
      name: 'Fresh Food',
      calories: 200,
      protein: 15,
      carbs: 25,
      fat: 5,
    };

    it('should return cached item when available', async () => {
      // Arrange
      (mockCache.get as any).mockResolvedValue(cachedFoodItem);

      // Act
      const result = await lookupService.lookupBarcode(testBarcode);

      // Assert
      expect(result).toEqual(cachedFoodItem);
      expect(mockCache.get).toBeCalledWith(`food_${testBarcode}`);
      expect(mockFoodDatabase.fetchFoodItem).not.toBeCalled();
    });

    it('should fetch from external API and cache on miss', async () => {
      // Arrange
      (mockCache.get as any).mockResolvedValue(null);
      (mockFoodDatabase.fetchFoodItem as any).mockResolvedValue(freshFoodItem);

      // Act
      const result = await lookupService.lookupBarcode(testBarcode);

      // Assert
      expect(result).toEqual(freshFoodItem);
      expect(mockFoodDatabase.fetchFoodItem).toBeCalledWith(testBarcode);
      expect(mockCache.set).toBeCalledWith(
        `food_${testBarcode}`,
        freshFoodItem,
        86400, // TTL (1 day)
      );
    });

    it('should throw error when barcode not found in DB', async () => {
      // Arrange
      (mockCache.get as any).mockResolvedValue(null);
      (mockFoodDatabase.fetchFoodItem as any).mockRejectedValue(new Error('Barcode not found'));

      // Act & Assert
      await expect(lookupService.lookupBarcode('INVALID_BARCODE')).rejects.toThrow(
        'Barcode not found',
      );
      expect(mockCache.set).not.toBeCalled();
    });

    it('should return partial nutrient data with defaults', async () => {
      // Arrange
      const partialFoodItem = {
        barcode: testBarcode,
        name: 'Partial Food',
        calories: 300,
        // Missing macros
      } as any;
      (mockCache.get as any).mockResolvedValue(null);
      (mockFoodDatabase.fetchFoodItem as any).mockResolvedValue(partialFoodItem);

      // Act
      const result = await lookupService.lookupBarcode(testBarcode);

      // Assert
      expect(result).toEqual({
        ...partialFoodItem,
        protein: 0, // Default to 0
        carbs: 0,
        fat: 0,
      });
    });
  });
});
