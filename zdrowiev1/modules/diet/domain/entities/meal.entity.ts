export interface MealProduct {
  name: string;
  quantity: number; // grams
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  productId?: string;
  barcode?: string;
}

export interface Meal {
  id: string;
  userId: string;
  name: string;
  consumedAt: Date;
  products: MealProduct[];
}

export interface NutritionValue {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}
