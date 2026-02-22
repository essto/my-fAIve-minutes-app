import { Anomaly, DietMetrics, HealthMetrics } from '../types/visualization.types';

export class DashboardService {
  private static readonly HEALTH_WEIGHT = 0.6;
  private static readonly DIET_WEIGHT = 0.4;

  static calculateHealthScore(healthData: HealthMetrics, dietData: DietMetrics): number {
    if (!healthData) return 0;

    const healthSubscore = this.calculateHealthSubscore(healthData);
    const dietSubscore = dietData ? this.calculateDietSubscore(dietData) : 0;

    return Math.round(healthSubscore * this.HEALTH_WEIGHT + dietSubscore * this.DIET_WEIGHT);
  }

  static detectAnomalies(healthData: Partial<HealthMetrics>): Anomaly[] {
    const anomalies: Anomaly[] = [];

    if (healthData.heartRate) {
      if (healthData.heartRate > 150) {
        anomalies.push({
          metric: 'heartRate',
          value: healthData.heartRate,
          severity: 'high',
          message: 'Tętno spoczynkowe powyżej 150 BPM!',
        });
      } else if (healthData.heartRate < 40) {
        anomalies.push({
          metric: 'heartRate',
          value: healthData.heartRate,
          severity: 'high',
          message: 'Bradykardia: tętno poniżej 40 BPM.',
        });
      }
    }

    return anomalies;
  }

  private static calculateHealthSubscore(health: HealthMetrics): number {
    let score = 0;

    // Heart rate (60-100 is ideal)
    if (health.heartRate >= 60 && health.heartRate <= 100) score += 25;
    else if (health.heartRate > 100 && health.heartRate < 120) score += 15;

    // Steps (target 10k)
    score += Math.min(25, (health.steps / 10000) * 25);

    // Oxygen (95-100)
    score += Math.min(25, Math.max(0, (health.oxygenSaturation - 90) * 2.5));

    // Blood Pressure
    if (health.bloodPressure) {
      const [sys, dia] = health.bloodPressure;
      if (sys <= 120 && dia <= 80) score += 25;
      else if (sys <= 140 && dia <= 90) score += 15;
    }

    return score;
  }

  private static calculateDietSubscore(diet: DietMetrics): number {
    let score = 0;

    // Calorie balance (mock target 2000)
    const calDiff = Math.abs(diet.calories - 2000);
    score += Math.max(0, 40 - calDiff / 10);

    // Protein (mock target 100g)
    score += Math.min(30, (diet.protein / 100) * 30);

    // Water (mock target 2000ml)
    score += Math.min(30, (diet.water / 2000) * 30);

    return Math.min(100, score);
  }
}
