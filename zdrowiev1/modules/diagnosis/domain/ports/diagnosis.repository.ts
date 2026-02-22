import { SymptomReport, TriageResult } from '../entities/symptom.entity';

export interface DiagnosisRepository {
  saveReport(report: SymptomReport): Promise<SymptomReport>;
  saveTriageResult(result: TriageResult): Promise<TriageResult>;
  findReportsByUserId(userId: string): Promise<SymptomReport[]>;
  findTriageResultByReportId(reportId: string): Promise<TriageResult | null>;
}
