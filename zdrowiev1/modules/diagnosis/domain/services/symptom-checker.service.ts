export class SymptomCheckerService {
  private readonly conditions = [
    { name: 'Grypa', symptoms: ['gorączka', 'ból głowy', 'dreszcze'] },
    { name: 'Przeziębienie', symptoms: ['katar', 'kaszel', 'lekka gorączka'] },
  ];

  async matchConditions(symptoms: any[]): Promise<any[]> {
    for (const s of symptoms) {
      if (!s.nazwa || !s.obszar || !s.intensywność || !s.czasTrwania) {
        throw new Error('InvalidSymptomDataError');
      }
    }

    const inputSymptomNames = symptoms.map((s) => s.nazwa.toLowerCase());

    return this.conditions
      .filter((condition) =>
        condition.symptoms.some((s) => inputSymptomNames.includes(s.toLowerCase())),
      )
      .map((c) => ({ name: c.name }));
  }
}
