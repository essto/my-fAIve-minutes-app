import { Module } from '@nestjs/common';
import { WeightModule } from '@modules/weight/weight.module';
import { HeartRateModule } from '@modules/heart-rate/heart-rate.module';
import { SleepModule } from '@modules/sleep/sleep.module';
import { DietModule } from '@modules/diet/diet.module';
import { DiagnosisModule } from '@modules/diagnosis/diagnosis.module';

@Module({
  imports: [WeightModule, HeartRateModule, SleepModule, DietModule, DiagnosisModule],
  controllers: [],
  providers: [],
})
export class AppModule {}
