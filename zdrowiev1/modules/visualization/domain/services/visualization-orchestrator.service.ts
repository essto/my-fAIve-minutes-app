import { Injectable } from '@nestjs/common';
import { ChartConfigService } from './chart-config.service';
import { DashboardService } from './dashboard.service';
import { ExportService } from './export.service';

@Injectable()
export class VisualizationOrchestrator {
  constructor() {}

  async getDashboardData(userId: string, healthData: any, dietData: any): Promise<any> {
    const healthScore = DashboardService.calculateHealthScore(healthData, dietData);
    const anomalies = DashboardService.detectAnomalies(healthData);

    const chartConfigs = {
      weightTrend: ChartConfigService.generateConfig('line', healthData.weightHistory || []),
      activityHeatmap: ChartConfigService.generateConfig('heatmap', healthData.activityData || []),
    };

    return {
      userId,
      healthScore,
      anomalies,
      charts: chartConfigs,
      timestamp: new Date().toISOString(),
    };
  }

  async exportCsv(userId: string, data: any[], headers: string[]): Promise<string> {
    ExportService.validateForCsv(data, headers);
    // In a real app, use json2csv here
    return 'mock,csv,data';
  }
}
