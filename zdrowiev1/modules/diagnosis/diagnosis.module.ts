import { Module } from '@nestjs/common';
import { DiagnosisController } from './application/controllers/diagnosis.controller';
import { DiagnosisService } from './domain/services/diagnosis.service';
import { DrizzleDiagnosisRepository } from './infrastructure/database/repository/drizzle-diagnosis.repository';

@Module({
  controllers: [DiagnosisController],
  providers: [
    {
      provide: 'DIAGNOSIS_SERVICE',
      useFactory: () => {
        const repository = new DrizzleDiagnosisRepository();
        return new DiagnosisService(repository);
      },
    },
  ],
  exports: ['DIAGNOSIS_SERVICE'],
})
export class DiagnosisModule {}
