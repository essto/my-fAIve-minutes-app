import { FoodItem } from '../entities/food-item.entity';

export interface FoodDatabasePort {
  fetchFoodItem(barcode: string): Promise<FoodItem>;
}
