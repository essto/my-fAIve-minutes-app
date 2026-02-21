import { Module } from '@nestjs/common';
import { DietController } from './application/controllers/diet.controller';
import { DietService } from './domain/services/diet.service';
import { BarcodeLookupService } from './domain/services/barcode-lookup.service';
import { DrizzleDietRepository } from './infrastructure/database/repository/drizzle-diet.repository';
import { OpenFoodFactsAdapter } from './infrastructure/external-api/open-food-facts.adapter';
import { RedisCacheAdapter } from './infrastructure/cache/redis-cache.adapter';

@Module({
  controllers: [DietController],
  providers: [
    {
      provide: 'DIET_SERVICE',
      useFactory: () => {
        const repository = new DrizzleDietRepository();
        return new DietService(repository);
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
  exports: ['DIET_SERVICE', 'BARCODE_SERVICE'],
})
export class DietModule {}
