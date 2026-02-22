import { describe, it, expect } from 'vitest';
import { TriageEngine } from '../services/triage-engine';
import { Symptom } from '../entities/symptom.entity';

describe('TriageEngine', () => {
  const engine = new TriageEngine();

  it('should return HIGH risk for severe and long duration symptoms', () => {
    const symptoms: Symptom[] = [{ name: 'Chest Pain', severity: 9, durationHours: 2 }];
    const result = engine.evaluate(symptoms);
    expect(result.riskLevel).toBe('HIGH');
    expect(result.recommendation).toContain('SOR');
  });

  it('should return MEDIUM risk for moderate symptoms', () => {
    const symptoms: Symptom[] = [{ name: 'Fever', severity: 6, durationHours: 24 }];
    const result = engine.evaluate(symptoms);
    expect(result.riskLevel).toBe('MEDIUM');
    expect(result.recommendation).toContain('lekarz');
  });

  it('should return LOW risk for mild symptoms', () => {
    const symptoms: Symptom[] = [{ name: 'Runny Nose', severity: 2, durationHours: 48 }];
    const result = engine.evaluate(symptoms);
    expect(result.riskLevel).toBe('LOW');
    expect(result.recommendation.toLowerCase()).toContain('odpoczynek');
  });

  it('should handle multiple symptoms by picking the highest risk', () => {
    const symptoms: Symptom[] = [
      { name: 'Chest Pain', severity: 4, durationHours: 1 },
      { name: 'Shortness of breath', severity: 8, durationHours: 1 },
    ];
    const result = engine.evaluate(symptoms);
    expect(result.riskLevel).toBe('HIGH');
  });
});
