import { Injectable } from '@nestjs/common';
import { ChartConfigService } from './chart-config.service';
import { DashboardService } from './dashboard.service';
import { DashboardData, DietMetrics, HealthMetrics } from '../types/visualization.types';

@Injectable()
export class VisualizationOrchestrator {
  constructor() {}

  async getDashboardData(
    userId: string,
    healthData: HealthMetrics & { weightHistory?: any[] },
    dietData: DietMetrics,
  ): Promise<DashboardData> {
    const healthScore = DashboardService.calculateHealthScore(healthData, dietData);
    const anomalies = DashboardService.detectAnomalies(healthData);

    const chartConfigs = {
      weightTrend: ChartConfigService.generateConfig(
        'line',
        healthData.weightHistory || [],
        'light',
      ),
      activityHeatmap: ChartConfigService.generateConfig(
        'heatmap',
        [], // To be integrated with real activity data
        'light',
      ),
    };

    return {
      userId,
      healthScore,
      anomalies,
      charts: chartConfigs,
      timestamp: new Date().toISOString(),
    };
  }
}
