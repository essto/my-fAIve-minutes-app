import { SymptomReport, Diagnosis } from '@monorepo/shared-types';

export interface DiagnosisRepository {
  saveSymptomReport(report: Partial<SymptomReport>): Promise<SymptomReport>;
  saveDiagnosis(diagnosis: Partial<Diagnosis>): Promise<Diagnosis>;
  findReportsByUserId(userId: string): Promise<SymptomReport[]>;
  findDiagnosesByUserId(userId: string): Promise<Diagnosis[]>;
}
