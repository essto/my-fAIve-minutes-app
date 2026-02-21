import { Module } from '@nestjs/common';
import { DiagnosisController } from './application/controllers/diagnosis.controller';
import { DiagnosisService } from './domain/services/diagnosis.service';
import { SymptomCheckerService } from './domain/services/symptom-checker.service';
import { TriageService } from './domain/services/triage.service';
import { ReportGeneratorService } from './domain/services/report-generator.service';
import { DrizzleDiagnosisRepository } from './infrastructure/database/repository/drizzle-diagnosis.repository';
import { db } from '../shared/database/src/index';

@Module({
  controllers: [DiagnosisController],
  providers: [
    SymptomCheckerService,
    TriageService,
    ReportGeneratorService,
    {
      provide: 'DIAGNOSIS_SERVICE',
      useFactory: (
        repository: DrizzleDiagnosisRepository,
        symptomChecker: SymptomCheckerService,
        triage: TriageService,
        reportGenerator: ReportGeneratorService,
      ) => {
        return new DiagnosisService(repository, symptomChecker, triage, reportGenerator);
      },
      inject: [
        DrizzleDiagnosisRepository,
        SymptomCheckerService,
        TriageService,
        ReportGeneratorService,
      ],
    },
    {
      provide: DrizzleDiagnosisRepository,
      useFactory: () => new DrizzleDiagnosisRepository(db),
    },
  ],
  exports: ['DIAGNOSIS_SERVICE'],
})
export class DiagnosisModule {}
