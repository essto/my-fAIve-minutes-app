export class ChartConfigService {
  static generateConfig(type: string, data: any[]): any {
    const supportedTypes = [
      'line',
      'area',
      'bar',
      'radar',
      'gauge',
      'heatmap',
      'sparkline',
      'candlestick',
    ];

    if (!supportedTypes.includes(type)) {
      throw new Error(`Nieobsługiwany typ wykresu: ${type}`);
    }

    if (type === 'line') {
      return {
        type: 'line',
        data: data.map((d) => ({
          x: d.timestamp,
          y: d.steps,
        })),
      };
    }

    if (type === 'heatmap') {
      // Fill missing values for 24 hours
      const filledData = [...data];
      const day = data[0]?.day || 'Mon';
      for (let hour = 0; hour < 24; hour++) {
        if (!filledData.find((d) => d.day === day && d.hour === hour)) {
          filledData.push({ day, hour, value: 0 });
        }
      }
      return { type: 'heatmap', data: filledData };
    }

    return { type, data };
  }
}
