import { DiagnosisRepository } from '../ports/diagnosis.repository';
import { SymptomReport, Diagnosis } from '@monorepo/shared-types';
import { SymptomCheckerService } from './symptom-checker.service';
import { TriageService } from './triage.service';
import { ReportGeneratorService } from './report-generator.service';

export class DiagnosisService {
  constructor(
    private readonly repository: DiagnosisRepository,
    private readonly symptomChecker: SymptomCheckerService,
    private readonly triage: TriageService,
    private readonly reportGenerator: ReportGeneratorService,
  ) {}

  async reportSymptoms(
    userId: string,
    data: { description: string; severity: number; history?: string[]; symptoms?: any[] },
  ): Promise<Diagnosis> {
    const report = await this.repository.saveSymptomReport({ userId, ...data });

    // Use SymptomChecker for condition matching
    const symptoms = data.symptoms || [{ nazwa: data.description, intensywność: data.severity }];
    const matches = await this.symptomChecker.matchConditions(symptoms);
    const diagnosisResult = matches[0]?.name || 'Inconclusive report';

    // Use Triage for risk assessment
    const history = data.history || [];
    const triageLevel = await this.triage.evaluateRisk(symptoms, history);

    return this.repository.saveDiagnosis({
      userId,
      symptomReportId: report.id,
      result: diagnosisResult,
      confidence: matches[0] ? 0.85 : 0.5,
      recommendations: this.getRecommendations(diagnosisResult, triageLevel),
    });
  }

  private getRecommendations(result: string, triage: string): string[] {
    const recommendations: string[] = [];
    if (triage === 'RED') recommendations.push('UDAJ SIĘ NATYCHMIAST NA SOR (RED ALERT)');
    if (triage === 'YELLOW') recommendations.push('Skontaktuj się z lekarzem w ciągu 24h');

    if (result === 'Grypa') recommendations.push('Odpoczywaj', 'Pij dużo płynów');
    return recommendations;
  }

  async generateReport(userId: string, diagnosisId: string): Promise<Buffer> {
    const diagnoses = await this.repository.findDiagnosesByUserId(userId);
    const diagnosis = diagnoses.find((d) => d.id === diagnosisId);
    if (!diagnosis) throw new Error('DiagnosisNotFound');

    // Simplified data for report generation demo
    return this.reportGenerator.generatePdf({
      userId,
      symptoms: [{ nazwa: diagnosis.result, czasTrwania: '2 dni' }],
      triageResult: 'UNKNOWN', // In a real app we'd fetch this from the repo
    });
  }

  async getHistory(userId: string): Promise<Diagnosis[]> {
    return this.repository.findDiagnosesByUserId(userId);
  }
}
