import { Module } from '@nestjs/common';
import { SleepController } from './application/controllers/sleep.controller';
import { SleepService } from './domain/services/sleep.service';
import { DrizzleSleepRepository } from './infrastructure/database/repository/drizzle-sleep.repository';

@Module({
  controllers: [SleepController],
  providers: [
    {
      provide: 'SLEEP_SERVICE',
      useFactory: () => {
        const repository = new DrizzleSleepRepository();
        return new SleepService(repository);
      },
    },
  ],
  exports: ['SLEEP_SERVICE'],
})
export class SleepModule {}
