import { DiagnosisRepository } from '../../domain/ports/diagnosis.repository';
import { Symptom, SymptomReport, TriageResult } from '../../domain/entities/symptom.entity';
import { TriageEngine } from '../../domain/services/triage-engine';

export class SymptomCheckerService {
  private readonly triageEngine = new TriageEngine();

  constructor(private readonly repository: DiagnosisRepository) {}

  async reportSymptoms(
    userId: string,
    symptoms: Symptom[],
  ): Promise<{ report: SymptomReport; triage: TriageResult }> {
    const report: SymptomReport = {
      id: crypto.randomUUID(),
      userId,
      createdAt: new Date(),
      symptoms,
    };

    const savedReport = await this.repository.saveReport(report);

    const evaluation = this.triageEngine.evaluate(symptoms);

    const triage: TriageResult = {
      id: crypto.randomUUID(),
      reportId: savedReport.id,
      riskLevel: evaluation.riskLevel,
      recommendation: evaluation.recommendation,
    };

    const savedTriage = await this.repository.saveTriageResult(triage);

    return {
      report: savedReport,
      triage: savedTriage,
    };
  }

  async getUserReports(userId: string): Promise<SymptomReport[]> {
    return this.repository.findReportsByUserId(userId);
  }

  async getReportTriage(reportId: string): Promise<TriageResult | null> {
    return this.repository.findTriageResultByReportId(reportId);
  }
}
