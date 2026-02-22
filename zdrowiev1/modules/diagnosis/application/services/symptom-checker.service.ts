import { DiagnosisRepository } from '../../domain/ports/diagnosis.repository';
import { Symptom, SymptomReport, TriageResult } from '../../domain/entities/symptom.entity';
import { TriageEngine } from '../../domain/services/triage-engine';
import { ReportGeneratorService } from '../../domain/services/report-generator.service';

export class SymptomCheckerService {
  private readonly triageEngine = new TriageEngine();
  private readonly reportGenerator = new ReportGeneratorService();

  constructor(private readonly repository: DiagnosisRepository) {}

  async reportSymptoms(
    userId: string,
    symptoms: Symptom[],
    aiHeuristic?: (symptoms: Symptom[]) => any,
  ): Promise<{ report: SymptomReport; triage: TriageResult; pdf?: Buffer }> {
    const report: SymptomReport = {
      id: crypto.randomUUID(),
      userId,
      createdAt: new Date(),
      symptoms,
    };

    const savedReport = await this.repository.saveReport(report);

    const evaluation = this.triageEngine.evaluate(symptoms, aiHeuristic);

    const triage: TriageResult = {
      id: crypto.randomUUID(),
      reportId: savedReport.id,
      riskLevel: evaluation.riskLevel,
      recommendation: evaluation.recommendation,
    };

    const savedTriage = await this.repository.saveTriageResult(triage);

    // Dynamic PDF generation - can be moved to background job later
    const pdf = await this.reportGenerator.generatePdf({
      userId,
      reportId: savedReport.id,
      symptoms,
      triageResult: evaluation.riskLevel,
      healthHistory: [], // Would fetch from health module in full implementation
      locale: 'pl',
    });

    return {
      report: savedReport,
      triage: savedTriage,
      pdf,
    };
  }

  async getUserReports(userId: string): Promise<SymptomReport[]> {
    return this.repository.findReportsByUserId(userId);
  }

  async getReportTriage(reportId: string): Promise<TriageResult | null> {
    return this.repository.findTriageResultByReportId(reportId);
  }
}
