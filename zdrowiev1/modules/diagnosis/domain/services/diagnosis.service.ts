import { DiagnosisRepository } from '../ports/diagnosis.repository';
import { SymptomReport, Diagnosis } from '@monorepo/shared-types';

export class DiagnosisService {
  constructor(private readonly repository: DiagnosisRepository) {}

  async reportSymptoms(
    userId: string,
    data: { description: string; severity: number },
  ): Promise<Diagnosis> {
    const report = await this.repository.saveSymptomReport({ userId, ...data });

    // Mock diagnosis logic
    const diagnosisResult = this.generateMockDiagnosis(data.description);

    return this.repository.saveDiagnosis({
      userId,
      symptomReportId: report.id,
      result: diagnosisResult.result,
      confidence: diagnosisResult.confidence,
      recommendations: diagnosisResult.recommendations,
    });
  }

  private generateMockDiagnosis(description: string): {
    result: string;
    confidence: number;
    recommendations: string[];
  } {
    const desc = description.toLowerCase();
    if (desc.includes('headache')) {
      return {
        result: 'Tension Headache',
        confidence: 0.85,
        recommendations: [
          'Rest in a quiet room',
          'Hydrate',
          'Consider over-the-counter pain relief',
        ],
      };
    }
    if (desc.includes('fever') || desc.includes('cough')) {
      return {
        result: 'Common Cold / Flu',
        confidence: 0.7,
        recommendations: ['Stay warm', 'Drink plenty of fluids', 'Monitor temperature'],
      };
    }
    return {
      result: 'Inconclusive report',
      confidence: 0.5,
      recommendations: ['Consult with a professional if symptoms persist'],
    };
  }

  async getHistory(userId: string): Promise<Diagnosis[]> {
    return this.repository.findDiagnosesByUserId(userId);
  }
}
