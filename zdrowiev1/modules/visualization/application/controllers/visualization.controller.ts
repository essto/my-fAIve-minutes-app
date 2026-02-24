import { Controller, Get, Post, Body, Req } from '@nestjs/common';
import { VisualizationOrchestrator } from '../../domain/services/visualization-orchestrator.service';
import { ExportService } from '../../domain/services/export.service';

@Controller('visualization')
export class VisualizationController {
  constructor(private readonly orchestrator: VisualizationOrchestrator) {}

  @Get('dashboard')
  async getDashboard(@Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';

    // Generate mock data matching the frontend DashboardData interface
    const now = new Date();
    const dates7 = Array.from({ length: 7 }, (_, i) => {
      const d = new Date(now);
      d.setDate(d.getDate() - (6 - i));
      return d.toISOString().split('T')[0];
    });

    return {
      healthScore: 85,
      anomalies: [
        {
          metric: 'Tętno',
          value: 95,
          severity: 'medium',
          message: 'Podwyższone tętno spoczynkowe - monitoruj przez kolejne dni',
        },
        {
          metric: 'Sen',
          value: 5.5,
          severity: 'low',
          message: 'Krótki sen ostatniej nocy - zadbaj o regularny harmonogram',
        },
      ],
      charts: {
        healthTrend: {
          type: 'line' as const,
          data: dates7.map((date) => ({
            label: date,
            value: 78 + Math.round(Math.random() * 10),
          })),
          colors: ['#6366f1'],
        },
        activityRings: {
          type: 'radar' as const,
          data: [
            { label: 'Ruch', value: 75 },
            { label: 'Sen', value: 90 },
            { label: 'Dieta', value: 60 },
            { label: 'Nawodnienie', value: 80 },
            { label: 'Tętno', value: 85 },
          ],
          colors: ['#10b981'],
        },
        sleepQuality: {
          type: 'bar' as const,
          data: dates7.map((date) => ({
            label: date,
            value: +(6 + Math.random() * 2.5).toFixed(1),
          })),
          colors: ['#8b5cf6'],
        },
      },
    };
  }

  @Get('health-details')
  async getHealthDetails(@Req() req: any) {
    // Mock health detail data for the Health page
    const now = new Date();
    const generateDates = (days: number) =>
      Array.from({ length: days }, (_, i) => {
        const d = new Date(now);
        d.setDate(d.getDate() - (days - 1 - i));
        return d.toISOString().split('T')[0];
      });

    const dates7 = generateDates(7);
    const dates90 = generateDates(90);

    return {
      metrics: {
        heartRate: { current: 72, avg7d: 70, min: 58, max: 95 },
        sleep: { lastNight: '7h 32min', avg7d: '7h 15min', quality: 'Dobra' },
        weight: { current: 78.5, change30d: -1.2, bmi: 24.1 },
      },
      charts: {
        heartRateHistory: {
          type: 'line' as const,
          data: dates7.map((date) => ({
            label: date,
            value: 65 + Math.round(Math.random() * 20),
          })),
          colors: ['#6366f1'],
        },
        sleepHistory: {
          type: 'bar' as const,
          data: dates7.map((date) => ({
            label: date,
            value: +(6 + Math.random() * 2.5).toFixed(1),
          })),
          colors: ['#8b5cf6'],
        },
        weightHistory: {
          type: 'area' as const,
          data: dates90
            .filter((_, i) => i % 3 === 0)
            .map((date, i) => ({
              label: date,
              value: +(80 - i * 0.05 + (Math.random() - 0.5)).toFixed(1),
            })),
          colors: ['#10b981'],
        },
      },
    };
  }

  @Post('export/csv')
  async exportCsv(@Body() body: { data: any[]; headers: string[] }) {
    return ExportService.toCSV(body.data, body.headers);
  }
}
