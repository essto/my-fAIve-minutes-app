import { ChartConfig, ChartDataPoint, ChartType, ThemeMode } from '../types/visualization.types';

export class ChartConfigService {
  private static readonly PALETTES = {
    light: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'],
    dark: ['#60a5fa', '#34d399', '#fbbf24', '#f87171', '#a78bfa'],
  };

  static generateConfig(
    type: ChartType,
    data: ChartDataPoint[],
    theme: ThemeMode = 'light',
  ): ChartConfig {
    const supportedTypes: ChartType[] = [
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

    if (!supportedTypes.includes(type)) {
      throw new Error(`Nieobsługiwany typ wykresu: ${type}`);
    }

    let processedData = data;

    if (type === 'heatmap') {
      processedData = this.fillHeatmapData(data);
    }

    return {
      type,
      data: processedData,
      theme,
      options: {
        responsive: true,
        animations: true,
        colors: this.PALETTES[theme],
      },
    };
  }

  private static fillHeatmapData(data: ChartDataPoint[]): ChartDataPoint[] {
    const filled: ChartDataPoint[] = [];
    const existing = new Map(data.map((d) => [new Date(d.timestamp).getUTCHours(), d.value]));

    for (let hour = 0; hour < 24; hour++) {
      filled.push({
        timestamp: hour,
        hour: hour,
        value: existing.get(hour) || 0,
      } as any);
    }
    return filled;
  }
}
