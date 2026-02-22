import { Module } from '@nestjs/common';
import { DiagnosisController } from './application/controllers/diagnosis.controller';
import { SymptomCheckerService } from './application/services/symptom-checker.service';
import { DrizzleDiagnosisRepository } from './infrastructure/database/repository/drizzle-diagnosis.repository';

@Module({
  controllers: [DiagnosisController],
  providers: [
    {
      provide: 'SYMPTOM_SERVICE',
      useFactory: () => {
        const repository = new DrizzleDiagnosisRepository();
        return new SymptomCheckerService(repository);
      },
    },
  ],
  exports: ['SYMPTOM_SERVICE'],
})
export class DiagnosisModule {}
