import { Module } from '@nestjs/common';
import { WeightModule } from '@monorepo/weight';
import { HeartRateModule } from '@monorepo/heart-rate';
import { SleepModule } from '@monorepo/sleep';
import { DietModule } from '@monorepo/diet';
import { DiagnosisModule } from '@monorepo/diagnosis';
import { AuthModule } from '@monorepo/auth';
import { UserModule } from '@monorepo/user';
import { SeedModule } from '@monorepo/database';
import { NotificationsModule } from '../../../modules/notifications/notifications.module';

@Module({
  imports: [
    AuthModule,
    UserModule,
    SeedModule,
    NotificationsModule,
    WeightModule,
    HeartRateModule,
    SleepModule,
    DietModule,
    DiagnosisModule,
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}
