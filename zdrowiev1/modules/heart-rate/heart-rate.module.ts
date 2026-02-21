import { Module } from '@nestjs/common';
import { HeartRateController } from './application/controllers/heart-rate.controller';
import { HeartRateService } from './domain/services/heart-rate.service';
import { DrizzleHeartRateRepository } from './infrastructure/database/repository/drizzle-heart-rate.repository';

@Module({
  controllers: [HeartRateController],
  providers: [
    {
      provide: 'HEART_RATE_SERVICE',
      useFactory: () => {
        const repository = new DrizzleHeartRateRepository();
        return new HeartRateService(repository);
      },
    },
  ],
  exports: ['HEART_RATE_SERVICE'],
})
export class HeartRateModule {}
