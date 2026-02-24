import { Symptom, TriageResult } from '../entities/symptom.entity';

export class TriageEngine {
  evaluate(
    symptoms: Symptom[],
    aiHeuristic?: (symptoms: Symptom[]) => 'LOW' | 'MEDIUM' | 'HIGH' | undefined,
  ): Omit<TriageResult, 'id' | 'reportId'> {
    // 1. Check AI Heuristic first
    const aiResult = aiHeuristic?.(symptoms);
    if (aiResult) {
      return {
        riskLevel: aiResult,
        recommendation: this.getRecommendationForLevel(aiResult),
      };
    }

    // 2. Fallback to standard heuristic logic
    let maxRisk: 'LOW' | 'MEDIUM' | 'HIGH' = 'LOW';
    let recommendations: string[] = [];

    for (const symptom of symptoms) {
      if (symptom.severity >= 8 || this.isEmergency(symptom)) {
        maxRisk = 'HIGH';
        recommendations = ['NATYCHMIAST skontaktuj się z SOR lub zadzwoń pod 112.'];
        break; // Zatrzymujemy analizę, mamy stan zagrożenia życia
      } else if (symptom.severity >= 5 || (symptom.severity >= 3 && symptom.durationHours > 24)) {
        maxRisk = 'MEDIUM';
        recommendations.push('Zalecana konsultacja z lekarzem pierwszego kontaktu w ciągu 24h.');
      } else {
        if (maxRisk === 'LOW') {
          recommendations.push('Odpoczynek i monitorowanie stanu zdrowia. Pij dużo płynów.');
        }
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

  private getRecommendationForLevel(level: 'LOW' | 'MEDIUM' | 'HIGH'): string {
    switch (level) {
      case 'HIGH':
        return 'NATYCHMIAST skontaktuj się z SOR lub zadzwoń pod 112.';
      case 'MEDIUM':
        return 'Zalecana konsultacja z lekarzem pierwszego kontaktu w ciągu 24h.';
      case 'LOW':
        return 'Odpoczynek i monitorowanie stanu zdrowia. Pij dużo płynów.';
    }
  }
}
