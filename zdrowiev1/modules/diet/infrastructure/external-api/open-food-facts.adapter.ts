import axios from 'axios';
import { FoodDatabasePort } from '../../domain/ports/food-database.port';
import { FoodItem } from '../../domain/entities/food-item.entity';

export class OpenFoodFactsAdapter implements FoodDatabasePort {
  private readonly baseUrl = 'https://world.openfoodfacts.org/api/v2/product';

  async fetchFoodItem(barcode: string): Promise<FoodItem> {
    try {
      const response = await axios.get(`${this.baseUrl}/${barcode}.json`);
      const data = response.data;

      if (data.status === 0) {
        throw new Error('Barcode not found');
      }

      const product = data.product;
      const nutriments = product.nutriments || {};

      return {
        barcode,
        name: product.product_name || 'Unknown Product',
        calories: nutriments['energy-kcal_100g'] || nutriments['energy-kcal'] || 0,
        protein: nutriments['proteins_100g'] || nutriments['proteins'] || 0,
        carbs: nutriments['carbohydrates_100g'] || nutriments['carbohydrates'] || 0,
        fat: nutriments['fat_100g'] || nutriments['fat'] || 0,
      };
    } catch (error: any) {
      if (error.message === 'Barcode not found') throw error;
      throw new Error(`Failed to fetch from Open Food Facts: ${error.message}`);
    }
  }
}
