import { describe, it, expect, beforeEach, vi } from 'vitest';
import { TriageEngine } from '../domain/services/triage-engine';
import { Symptom } from '../domain/entities/symptom.entity';

describe('TriageEngine AI Scenarios', () => {
  let triageEngine: TriageEngine;

  beforeEach(() => {
    triageEngine = new TriageEngine();
  });

  it('Scenariusz 1: should return HIGH risk when AI heuristic identifies a critical pattern', () => {
    const symptoms: Symptom[] = [
      { name: 'Silny ból w klatce piersiowej', severity: 7, durationHours: 1 } as any,
    ];

    // AI identifies this as critical even though severity is < 8
    const aiHeuristic = vi.fn().mockReturnValue('HIGH');

    const result = triageEngine.evaluate(symptoms, aiHeuristic);

    expect(result.riskLevel).toBe('HIGH');
    expect(aiHeuristic).toHaveBeenCalledWith(symptoms);
  });

  it('Scenariusz 2: should prioritize AI decision over raw severity thresholds', () => {
    const symptoms: Symptom[] = [{ name: 'Lekki kaszel', severity: 9, durationHours: 1 } as any];

    // Raw severity 9 would normally be HIGH, but AI says MEDIUM
    const aiHeuristic = vi.fn().mockReturnValue('MEDIUM');

    const result = triageEngine.evaluate(symptoms, aiHeuristic);

    expect(result.riskLevel).toBe('MEDIUM');
  });

  it('Scenariusz 3: should fallback to standard logic when AI returns null/undefined', () => {
    const symptoms: Symptom[] = [
      { name: 'Dziwne mrowienie', severity: 4, durationHours: 2 } as any,
    ];

    const aiHeuristic = vi.fn().mockReturnValue(undefined);

    const result = triageEngine.evaluate(symptoms, aiHeuristic);

    // Fallback: severity 4, low duration -> LOW risk
    expect(result.riskLevel).toBe('LOW');
  });

  it('Scenariusz 4: should handle batch processing of symptoms consistently with AI', () => {
    const symptoms: Symptom[] = Array(15).fill({ name: 'Symptom', severity: 2, durationHours: 1 });

    const aiHeuristic = vi.fn().mockReturnValue('HIGH');

    const result = triageEngine.evaluate(symptoms, aiHeuristic);

    expect(result.riskLevel).toBe('HIGH');
  });
});
