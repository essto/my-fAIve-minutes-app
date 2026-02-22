import { Symptom, TriageResult } from '../entities/symptom.entity';

export class TriageEngine {
  evaluate(symptoms: Symptom[]): Omit<TriageResult, 'id' | 'reportId'> {
    let maxRisk: 'LOW' | 'MEDIUM' | 'HIGH' = 'LOW';
    let recommendations: string[] = [];

    for (const symptom of symptoms) {
      if (symptom.severity >= 8 || this.isEmergency(symptom)) {
        maxRisk = 'HIGH';
        recommendations.push('NATYCHMIAST skontaktuj się z SOR lub zadzwoń pod 112.');
      } else if (symptom.severity >= 5 || (symptom.severity >= 3 && symptom.durationHours > 24)) {
        if (maxRisk !== 'HIGH') maxRisk = 'MEDIUM';
        recommendations.push('Zalecana konsultacja z lekarzem pierwszego kontaktu w ciągu 24h.');
      } else {
        recommendations.push('Odpoczynek i monitorowanie stanu zdrowia. Pij dużo płynów.');
      }
    }

    // Deduplicate and combine recommendations
    const uniqueRecs = Array.from(new Set(recommendations));

    return {
      riskLevel: maxRisk,
      recommendation: uniqueRecs.join(' '),
    };
  }

  private isEmergency(symptom: Symptom): boolean {
    const emergencyKeywords = [
      'chest pain',
      'shortness of breath',
      'unconscious',
      'ból w klatce',
      'duszność',
    ];
    return emergencyKeywords.some((key) => symptom.name.toLowerCase().includes(key));
  }
}
