export type ChartType =
  | 'line'
  | 'area'
  | 'bar'
  | 'radar'
  | 'gauge'
  | 'heatmap'
  | 'scatter'
  | 'progress_ring'
  | 'sparkline'
  | 'candlestick';

export type ThemeMode = 'light' | 'dark';

export interface ChartDataPoint {
  timestamp: string | number;
  value: number | [number, number];
  label?: string;
  metadata?: Record<string, any>;
}

export interface ChartConfig {
  type: ChartType;
  data: ChartDataPoint[];
  theme: ThemeMode;
  options: {
    responsive: boolean;
    animations: boolean;
    colors: string[];
    [key: string]: any;
  };
}

export interface HealthMetrics {
  heartRate: number;
  bloodPressure: [number, number];
  oxygenSaturation: number;
  steps: number;
  weight?: number;
}

export interface DietMetrics {
  calories: number;
  protein: number;
  fat: number;
  carbs: number;
  water: number;
}

export interface DashboardData {
  userId: string;
  healthScore: number;
  anomalies: Anomaly[];
  charts: Record<string, ChartConfig>;
  timestamp: string;
}

export interface Anomaly {
  metric: string;
  value: any;
  severity: 'low' | 'medium' | 'high';
  message: string;
}
