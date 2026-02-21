import { describe, it, expect, vi } from 'vitest';
import { DiagnosisService } from '../domain/services/diagnosis.service';
import { DiagnosisRepository } from '../domain/ports/diagnosis.repository';

describe('DiagnosisService', () => {
  const mockRepository: DiagnosisRepository = {
    saveSymptomReport: vi.fn(),
    saveDiagnosis: vi.fn(),
    findReportsByUserId: vi.fn(),
    findDiagnosesByUserId: vi.fn(),
  };

  const service = new DiagnosisService(mockRepository);

  it('generates a diagnosis for a headache', async () => {
    const mockReport = { id: 'r1', userId: 'u1', description: 'Headache', severity: 5 };
    const mockDiagnosis = { id: 'd1', userId: 'u1', result: 'Tension Headache', confidence: 0.85 };

    vi.mocked(mockRepository.saveSymptomReport).mockResolvedValue(mockReport as any);
    vi.mocked(mockRepository.saveDiagnosis).mockResolvedValue(mockDiagnosis as any);

    const result = await service.reportSymptoms('u1', { description: 'Headache', severity: 5 });

    expect(result.result).toBe('Tension Headache');
    expect(mockRepository.saveSymptomReport).toHaveBeenCalled();
    expect(mockRepository.saveDiagnosis).toHaveBeenCalledWith(
      expect.objectContaining({
        result: 'Tension Headache',
      }),
    );
  });

  it('handles unidentified symptoms with inconclusive result', async () => {
    const mockReport = { id: 'r2', userId: 'u1', description: 'Unknown thing', severity: 2 };
    vi.mocked(mockRepository.saveSymptomReport).mockResolvedValue(mockReport as any);
    vi.mocked(mockRepository.saveDiagnosis).mockImplementation((d) => Promise.resolve(d as any));

    const result = await service.reportSymptoms('u1', {
      description: 'Unknown thing',
      severity: 2,
    });
    expect(result.result).toBe('Inconclusive report');
  });
});
