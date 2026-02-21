export class TriageService {
  async evaluateRisk(symptoms: any[], history: string[]): Promise<'GREEN' | 'YELLOW' | 'RED'> {
    const symptomsNames = symptoms.map((s) => s.nazwa.toLowerCase());
    const maxIntensity = Math.max(...symptoms.map((s) => s.intensywność || 0));

    // TC2.1: ESCALATION FOR CRITICAL SYMPTOMS
    if (symptomsNames.includes('ból w klatce piersiowej') && maxIntensity >= 8) {
      return 'RED';
    }

    // TC2.2: HISTORY INFLUENCE
    if (symptomsNames.includes('duszności') && history.includes('asthma')) {
      return 'YELLOW';
    }

    // TC2.3: DURATION ESCALATION
    for (const s of symptoms) {
      if (s.intensywność <= 3 && s.czasTrwaniaDni >= 7) {
        return 'YELLOW';
      }
    }

    if (maxIntensity > 7) return 'YELLOW';

    return 'GREEN';
  }
}
