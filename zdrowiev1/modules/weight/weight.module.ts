import { Module } from '@nestjs/common';
import { WeightController } from './application/controllers/weight.controller';
import { WeightService } from './domain/services/weight.service';
import { DrizzleWeightRepository } from './infrastructure/database/repository/drizzle-weight.repository';
import { db, DATABASE_CONNECTION } from '@monorepo/database';

@Module({
  controllers: [WeightController],
  providers: [
    {
      provide: 'WEIGHT_SERVICE',
      useFactory: (db) => {
        const repository = new DrizzleWeightRepository(db);
        return new WeightService(repository);
      },
      inject: [DATABASE_CONNECTION],
    },
  ],
  exports: ['WEIGHT_SERVICE'],
})
export class WeightModule {}
