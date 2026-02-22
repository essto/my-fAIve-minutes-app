import { Module } from '@nestjs/common';
import { ActivityController } from './http/activity.controller';
import { ActivityService } from '../application/activity.service';
import { ActivityDrizzleRepository } from './database/activity.drizzle.repository';
import { DatabaseModule } from '@monorepo/database';

@Module({
  imports: [DatabaseModule],
  controllers: [ActivityController],
  providers: [
    ActivityService,
    {
      provide: 'IActivityRepository',
      useClass: ActivityDrizzleRepository,
    },
  ],
  exports: [ActivityService],
})
export class ActivityModule {}
