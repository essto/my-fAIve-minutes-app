import { describe, it, expect, beforeEach } from 'vitest';
import { ChartConfigService } from '../domain/services/chart-config.service';
import { DashboardService } from '../domain/services/dashboard.service';
import { ExportService } from '../domain/services/export.service';

describe('Visualization Module (Etap 6) - Unit Tests', () => {
  describe('ChartConfigService', () => {
    it('TC1.1: should throw error when chart type is invalid', () => {
      const invalidChartType = 'invalid';
      const sampleData = [{ date: '2023-10-01', value: 50 }];
      expect(() => ChartConfigService.generateConfig(invalidChartType, sampleData)).toThrow(
        'Nieobsługiwany typ wykresu: invalid',
      );
    });

    it('TC1.2: should return valid Recharts config for Line chart', () => {
      const data = [{ timestamp: '2023-10-01', steps: 8000 }];
      const config = ChartConfigService.generateConfig('line', data);
      expect(config.data[0].x).toBe('2023-10-01');
      expect(config.type).toBe('line');
    });

    it('TC1.3: should fill missing values for Heatmap', () => {
      const incompleteData = [{ day: 'Mon', hour: 12, value: 30 }];
      const config = ChartConfigService.generateConfig('heatmap', incompleteData);
      const found = config.data.find((d: any) => d.day === 'Mon' && d.hour === 14);
      expect(found.value).toBe(0);
    });
  });

  describe('DashboardService', () => {
    it('TC2.1: should calculate Health Score as weighted average (Health 60%, Diet 40%)', () => {
      const healthData = { score: 85 };
      const dietData = { score: 90 };
      const score = DashboardService.calculateHealthScore(healthData, dietData);
      expect(score).toBe(85 * 0.6 + 90 * 0.4); // 87
    });

    it('TC2.2: should flag anomaly for heart rate > 160', () => {
      const healthData = { heartRate: 170 };
      const anomalies = DashboardService.detectAnomalies(healthData);
      expect(anomalies).toContain('heartRate');
    });

    it('TC2.3: should return 0 if health data is missing', () => {
      const score = DashboardService.calculateHealthScore(null, { score: 90 });
      expect(score).toBe(0);
    });
  });

  describe('ExportService', () => {
    it('TC3.1: should reject CSV export with missing headers', () => {
      const invalidData = [{ value: 50 }];
      expect(() => ExportService.validateForCsv(invalidData, ['date', 'value'])).toThrow(
        'Brak wymaganych kolumn: date',
      );
    });

    it('TC3.2: should check for chart presence in PDF data', () => {
      const pdfData = { text: 'Summary', charts: [] };
      expect(() => ExportService.validateForPdf(pdfData)).toThrow(
        'Brak wymaganych wykresów w danych',
      );
    });

    it('TC3.3: should enforce ISO 8601 dates in CSV', () => {
      const data = [{ date: '2023/10/01', value: 50 }];
      ExportService.validateForCsv(data, ['date', 'value']);
      expect(data[0].date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    });
  });
});
