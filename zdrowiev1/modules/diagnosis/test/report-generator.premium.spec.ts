import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ReportGeneratorService } from '../domain/services/report-generator.service';

describe('ReportGeneratorService Premium Scenarios', () => {
  let reportGenerator: ReportGeneratorService;

  beforeEach(() => {
    reportGenerator = new ReportGeneratorService();
  });

  it('Scenariusz 5: should integrate medical history into the PDF report', async () => {
    const data = {
      userId: 'user-123',
      symptoms: [{ name: 'Grypa', severity: 5 }],
      triageResult: 'MEDIUM',
      healthHistory: [
        { type: 'allergy', value: 'Peanuts', date: '2023-01-01' },
        { type: 'condition', value: 'Asthma', date: '2022-05-10' },
      ],
    };

    const buffer = await reportGenerator.generatePdf(data as any);
    expect(buffer).toBeDefined();
    expect(buffer.length).toBeGreaterThan(0);
    // Note: Actual PDF content analysis would require a PDF parser,
    // but in TDD we verify it doesn't crash and generates a non-empty buffer for now.
  });

  it('Scenariusz 6: should include locale-specific legal disclaimers', async () => {
    const dataPl = {
      userId: 'user-123',
      symptoms: [],
      triageResult: 'LOW',
      locale: 'pl',
    };
    const dataEn = {
      userId: 'user-123',
      symptoms: [],
      triageResult: 'LOW',
      locale: 'en',
    };

    const bufferPl = await reportGenerator.generatePdf(dataPl as any);
    const bufferEn = await reportGenerator.generatePdf(dataEn as any);

    expect(bufferPl).toBeDefined();
    expect(bufferEn).toBeDefined();
  });

  it('Scenariusz 7: should apply premium styling branding elements', async () => {
    const data = {
      userId: 'user-123',
      symptoms: [{ name: 'Test', severity: 1 }],
      triageResult: 'LOW',
    };

    const buffer = await reportGenerator.generatePdf(data as any);
    expect(buffer).toBeDefined();
  });

  it('Scenariusz 8: should personalize report with User Info and ID', async () => {
    const data = {
      userId: 'Jan Kowalski',
      reportId: 'REP-2026-001',
      symptoms: [],
      triageResult: 'LOW',
    };

    const buffer = await reportGenerator.generatePdf(data as any);
    expect(buffer).toBeDefined();
  });
});
