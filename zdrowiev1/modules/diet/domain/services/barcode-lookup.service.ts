import { FoodDatabasePort } from '../ports/food-database.port';
import { CachePort } from '../ports/cache.port';
import { FoodItem } from '../entities/food-item.entity';

export class BarcodeLookupService {
  constructor(
    private readonly foodDb: FoodDatabasePort,
    private readonly cache: CachePort,
  ) {}

  async lookupBarcode(barcode: string): Promise<FoodItem> {
    if (!barcode) {
      throw new Error('Invalid barcode');
    }

    const cacheKey = `food_${barcode}`;
    const cached = await this.cache.get<FoodItem>(cacheKey);
    if (cached) {
      return cached;
    }

    const fresh = await this.foodDb.fetchFoodItem(barcode);

    // Ensure defaults for macros if missing
    const item: FoodItem = {
      ...fresh,
      protein: fresh.protein ?? 0,
      carbs: fresh.carbs ?? 0,
      fat: fresh.fat ?? 0,
    };

    await this.cache.set(cacheKey, item, 86400); // 1 day TTL
    return item;
  }
}
