import { Module } from '@nestjs/common';
import { DietController } from './application/controllers/diet.controller';
import { MealService } from './application/services/meal.service';
import { BarcodeLookupService } from './domain/services/barcode-lookup.service';
import { DrizzleMealRepository } from './infrastructure/database/repository/drizzle-meal.repository';
import { OpenFoodFactsAdapter } from './infrastructure/external-api/open-food-facts.adapter';
import { RedisCacheAdapter } from './infrastructure/cache/redis-cache.adapter';

@Module({
  controllers: [DietController],
  providers: [
    {
      provide: 'MEAL_SERVICE',
      useFactory: () => {
        const repository = new DrizzleMealRepository();
        return new MealService(repository);
      },
    },
    {
      provide: 'BARCODE_SERVICE',
      useFactory: () => {
        const foodDb = new OpenFoodFactsAdapter();
        const cache = new RedisCacheAdapter();
        return new BarcodeLookupService(foodDb, cache);
      },
    },
  ],
  exports: ['MEAL_SERVICE', 'BARCODE_SERVICE'],
})
export class DietModule {}
