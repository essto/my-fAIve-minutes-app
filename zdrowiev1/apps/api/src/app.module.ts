import { Module } from '@nestjs/common';
import { WeightModule } from '../../../modules/weight/weight.module';
import { HeartRateModule } from '../../../modules/heart-rate/heart-rate.module';
import { SleepModule } from '../../../modules/sleep/sleep.module';
import { DietModule } from '../../../modules/diet/diet.module';
import { DiagnosisModule } from '../../../modules/diagnosis/diagnosis.module';
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
