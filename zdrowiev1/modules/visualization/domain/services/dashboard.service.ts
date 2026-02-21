export class DashboardService {
  static calculateHealthScore(healthData: any, dietData: any): number {
    if (!healthData) return 0;

    const healthWeight = 0.6;
    const dietWeight = 0.4;

    const healthScore = healthData.score || 0;
    const dietScore = dietData?.score || 0;

    return healthScore * healthWeight + dietScore * dietWeight;
  }

  static detectAnomalies(healthData: any): string[] {
    const anomalies: string[] = [];
    if (healthData.heartRate > 160 || healthData.heartRate < 40) {
      anomalies.push('heartRate');
    }
    return anomalies;
  }
}
