import { Module } from '@nestjs/common';
import { ThrottlerModule, ThrottlerGuard } from '@nestjs/throttler';
import { APP_GUARD } from '@nestjs/core';
import { WeightModule } from '../../../modules/weight/weight.module';
import { HeartRateModule } from '../../../modules/heart-rate/heart-rate.module';
import { SleepModule } from '../../../modules/sleep/sleep.module';
import { DietModule } from '../../../modules/diet/diet.module';
import { DiagnosisModule } from '../../../modules/diagnosis/diagnosis.module';
import { AuthModule } from '@monorepo/auth';
import { UserModule } from '@monorepo/user';
import { SeedModule } from '@monorepo/database';
import { NotificationsModule } from '../../../modules/notifications/notifications.module';
import { ActivityModule } from '../../../modules/activity/src/infrastructure/activity.module';

@Module({
  imports: [
    ThrottlerModule.forRoot([
      { name: 'short', ttl: 1000, limit: 3 },   // 3 req/s
      { name: 'medium', ttl: 10000, limit: 20 }, // 20 req/10s
      { name: 'long', ttl: 60000, limit: 100 },  // 100 req/min
    ]),
    AuthModule,
    UserModule,
    SeedModule,
    NotificationsModule,
    ActivityModule,
    WeightModule,
    HeartRateModule,
    SleepModule,
    DietModule,
    DiagnosisModule,
  ],
  controllers: [],
  providers: [
    { provide: APP_GUARD, useClass: ThrottlerGuard },
  ],
})
export class AppModule { }
