import { describe, it, expect } from 'vitest';
import { ChartConfigService } from '../domain/services/chart-config.service';
import { DashboardService } from '../domain/services/dashboard.service';
import { VisualizationOrchestrator } from '../domain/services/visualization-orchestrator.service';
import { ExportService } from '../domain/services/export.service';
import { ChartType } from '../domain/types/visualization.types';

describe('Visualization Module (Etap 6) - Unit Tests', () => {
  describe('ChartConfigService', () => {
    it('TC1.1: should throw error when chart type is invalid', () => {
      const invalidChartType = 'invalid' as ChartType;
      const sampleData = [{ timestamp: '2023-10-01', value: 50 }];
      expect(() => ChartConfigService.generateConfig(invalidChartType, sampleData)).toThrow(
        'Nieobsługiwany typ wykresu: invalid',
      );
    });

    const chartTypes: ChartType[] = [
      'line',
      'area',
      'bar',
      'radar',
      'gauge',
      'heatmap',
      'scatter',
      'progress_ring',
      'sparkline',
      'candlestick',
    ];

    it.each(chartTypes)('TC1.2: should return valid config for chart type %s', (type) => {
      const data = [{ timestamp: '2023-10-01', value: 80 }];
      const config = ChartConfigService.generateConfig(type, data);
      expect(config.type).toBe(type);
      expect(config.data).toBeDefined();
    });

    it('TC1.3: should apply dark theme colors when specified', () => {
      const data = [{ timestamp: '10:00', value: 72 }];
      const config = ChartConfigService.generateConfig('line', data, 'dark');
      expect(config.theme).toBe('dark');
      expect(config.options.colors).toContain('#34d399'); // Example primary color for dark theme
    });

    it('TC1.4: should fill missing values for Heatmap (24h coverage)', () => {
      const incompleteData = [{ timestamp: '2023-10-01T12:00:00Z', value: 30 }];
      const config = ChartConfigService.generateConfig('heatmap', incompleteData);
      expect(config.data.length).toBe(24);
      const found = config.data.find((d: any) => d.hour === 14);
      expect(found).toBeDefined();
      expect(found?.value).toBe(0);
    });

    it('TC1.5: should handle Candlestick data format', () => {
      const candleData: [number, number] = [60, 80];
      const data = [{ timestamp: '2023-10-01', value: candleData }]; // min, max
      const config = ChartConfigService.generateConfig('candlestick', data);
      expect(Array.isArray(config.data[0].value)).toBe(true);
    });
  });

  describe('DashboardService', () => {
    it('TC2.1: should calculate complex Health Score correct weighted average', () => {
      const healthData = {
        heartRate: 75,
        bloodPressure: [120, 80] as [number, number],
        oxygenSaturation: 98,
        steps: 10000,
      };
      const dietData = {
        calories: 2200,
        protein: 150,
        fat: 70,
        carbs: 250,
        water: 2500,
      };

      const score = DashboardService.calculateHealthScore(healthData, dietData);
      expect(score).toBeGreaterThan(0);
      expect(score).toBeLessThanOrEqual(100);
    });

    it('TC2.2: should detect severe heart rate anomaly', () => {
      const healthData = { heartRate: 180 };
      const anomalies = DashboardService.detectAnomalies(healthData);
      expect(anomalies).toHaveLength(1);
      expect(anomalies[0].severity).toBe('high');
      expect(anomalies[0].metric).toBe('heartRate');
    });

    it('TC2.3: should return 0 for health score if data is missing', () => {
      const score = DashboardService.calculateHealthScore(null as any, null as any);
      expect(score).toBe(0);
    });
  });

  describe('VisualizationOrchestrator', () => {
    it('TC3.1: should aggregate dashboard data correctly', async () => {
      const orchestrator = new VisualizationOrchestrator();
      const healthData = {
        heartRate: 72,
        bloodPressure: [120, 80] as [number, number],
        oxygenSaturation: 98,
        steps: 8000,
        weightHistory: [{ timestamp: '2023-10-01', value: 75 }],
      };
      const dietData = {
        calories: 2000,
        protein: 100,
        fat: 65,
        carbs: 250,
        water: 2000,
      };

      const dashboard = await orchestrator.getDashboardData(
        'user-1',
        healthData as any,
        dietData as any,
      );

      expect(dashboard.userId).toBe('user-1');
      expect(dashboard.healthScore).toBeGreaterThan(0);
      expect(dashboard.charts.weightTrend).toBeDefined();
      expect(dashboard.charts.weightTrend.type).toBe('line');
    });
  });

  describe('ExportService', () => {
    it('TC4.1: should export simple data to CSV format', () => {
      const data = [
        { date: '2023-10-01', value: 75.5 },
        { date: '2023-10-02', value: 75.2 },
      ];
      const headers = ['date', 'value'];

      const csv = ExportService.toCSV(data, headers);
      expect(csv).toContain('date,value');
      expect(csv).toContain('2023-10-01,75.5');
    });

    it('TC4.2: should throw error for PDF export without data', async () => {
      await expect(ExportService.generatePDF(null as any)).rejects.toThrow(
        'Brak danych do wygenerowania raportu PDF',
      );
    });
  });
});
