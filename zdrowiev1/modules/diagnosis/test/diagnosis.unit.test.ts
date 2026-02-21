import { describe, it, expect, beforeEach, vi } from 'vitest';
import { SymptomCheckerService } from '../domain/services/symptom-checker.service';
import { TriageService } from '../domain/services/triage.service';

describe('Diagnosis Module (Etap 4) - Unit Tests', () => {
  describe('SymptomCheckerService', () => {
    let symptomChecker: SymptomCheckerService;

    beforeEach(() => {
      // Mock disease database would go here if needed
      symptomChecker = new SymptomCheckerService();
    });

    it('TC1.1: should match disease based on symptoms', async () => {
      const symptoms = [
        { nazwa: 'gorączka', obszar: 'głowa', intensywność: 7, czasTrwania: '2 dni' },
      ];
      const matches = await symptomChecker.matchConditions(symptoms);

      expect(matches).toContainEqual(expect.objectContaining({ name: 'Grypa' }));
    });

    it('TC1.2: should return empty list for unusual symptoms', async () => {
      const symptoms = [
        {
          nazwa: 'mrowienie w stopach',
          obszar: 'stopy',
          intensywność: 2,
          czasTrwania: '1 godzina',
        },
      ];
      const matches = await symptomChecker.matchConditions(symptoms);

      expect(matches).toHaveLength(0);
    });

    it('TC1.3: should throw error for incomplete symptom data', async () => {
      const symptoms = [{ nazwa: 'gorączka', obszar: 'głowa', intensywność: 7 }] as any;

      await expect(symptomChecker.matchConditions(symptoms)).rejects.toThrow(
        'InvalidSymptomDataError',
      );
    });
  });

  describe('TriageService', () => {
    let triageService: TriageService;

    beforeEach(() => {
      triageService = new TriageService();
    });

    it('TC2.1: should escalate to RED for critical symptoms (chest pain)', async () => {
      const symptoms = [{ nazwa: 'ból w klatce piersiowej', intensywność: 9 }];
      const history: string[] = [];

      const level = await triageService.evaluateRisk(symptoms, history);

      expect(level).toBe('RED');
    });

    it('TC2.2: should escalate to YELLOW for historical context (asthma + shortness of breath)', async () => {
      const symptoms = [{ nazwa: 'duszności', intensywność: 4 }];
      const history = ['asthma'];

      const level = await triageService.evaluateRisk(symptoms, history);

      expect(level).toBe('YELLOW');
    });

    it('TC2.3: should escalate to YELLOW for long duration despite low intensity', async () => {
      const symptoms = [{ nazwa: 'gorączka', intensywność: 3, czasTrwaniaDni: 7 }];
      const history: string[] = [];

      const level = await triageService.evaluateRisk(symptoms, history);

      expect(level).toBe('YELLOW');
    });
  });

  describe('ReportGeneratorService', () => {
    let reportGenerator: any; // Type will be ReportGeneratorService

    beforeEach(async () => {
      // Use dynamic import or just import if available
      try {
        const { ReportGeneratorService } =
          await import('../domain/services/report-generator.service');
        reportGenerator = new ReportGeneratorService();
      } catch (e) {
        // If file doesn't exist yet, we will mock or wait
      }
    });

    it('TC3.1: should throw error for missing user_id', async () => {
      const data = { symptoms: [] };
      await expect(reportGenerator.generatePdf(data as any)).rejects.toThrow(
        'InvalidReportDataError',
      );
    });

    it('TC3.2: should throw error for invalid temporal format', async () => {
      const data = {
        userId: '123',
        symptoms: [{ nazwa: 'gorączka', czasTrwania: 'dwa tygodnie' }],
      };
      await expect(reportGenerator.generatePdf(data as any)).rejects.toThrow(
        'InvalidDateFormatError',
      );
    });

    it('TC3.3: should generate minimalist report buffer', async () => {
      const data = {
        userId: '123',
        symptoms: [{ nazwa: 'gorączka', czasTrwania: '2 dni' }],
        triageResult: 'GREEN',
      };
      const buffer = await reportGenerator.generatePdf(data);

      expect(buffer).toBeDefined();
      expect(buffer.length).toBeGreaterThan(0);
    });
  });
});
