import { db } from '@monorepo/database';
import { meals, mealProducts } from '../../../../shared/database/src/drizzle/schema';
import { MealRepository } from '../../../domain/ports/diet.repository';
import { Meal } from '../../../domain/entities/meal.entity';
import { eq, and, gte } from 'drizzle-orm';

export class DrizzleMealRepository implements MealRepository {
  async save(meal: Meal): Promise<Meal> {
    return await db.transaction(async (tx) => {
      const [insertedMeal] = await tx
        .insert(meals)
        .values({
          id: meal.id,
          userId: meal.userId,
          name: meal.name,
          consumedAt: meal.consumedAt,
        })
        .onConflictDoUpdate({
          target: meals.id,
          set: { name: meal.name, consumedAt: meal.consumedAt },
        })
        .returning();

      // Clear existing products if updating
      await tx.delete(mealProducts).where(eq(mealProducts.mealId, meal.id));

      if (meal.products.length > 0) {
        await tx.insert(mealProducts).values(
          meal.products.map((p) => ({
            mealId: meal.id,
            name: p.name,
            barcode: p.barcode,
            quantity: p.quantity,
            calories: p.calories,
            protein: p.protein,
            carbs: p.carbs,
            fat: p.fat,
            productId: p.productId,
          })),
        );
      }

      return {
        ...insertedMeal,
        products: meal.products,
      };
    });
  }

  async findById(id: string): Promise<Meal | null> {
    const [meal] = await db.select().from(meals).where(eq(meals.id, id));
    if (!meal) return null;

    const products = await db.select().from(mealProducts).where(eq(mealProducts.mealId, id));

    return {
      ...meal,
      products: products.map((p) => ({
        ...p,
        productId: p.productId ?? undefined,
        barcode: p.barcode ?? undefined,
      })),
    };
  }

  async findByUserId(userId: string, date?: Date): Promise<Meal[]> {
    let query = db.select().from(meals).where(eq(meals.userId, userId));

    if (date) {
      const startOfDay = new Date(date);
      startOfDay.setHours(0, 0, 0, 0);
      query = db
        .select()
        .from(meals)
        .where(and(eq(meals.userId, userId), gte(meals.consumedAt, startOfDay)));
    }

    const results = await query;
    const mealsWithProducts = await Promise.all(
      results.map(async (meal) => {
        const products = await db
          .select()
          .from(mealProducts)
          .where(eq(mealProducts.mealId, meal.id));
        return {
          ...meal,
          products: products.map((p) => ({
            ...p,
            productId: p.productId ?? undefined,
            barcode: p.barcode ?? undefined,
          })),
        };
      }),
    );

    return mealsWithProducts;
  }
}
