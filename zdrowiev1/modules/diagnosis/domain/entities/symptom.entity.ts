export interface Symptom {
  id?: string;
  reportId?: string;
  name: string;
  severity: number; // 1-10
  durationHours: number;
}

export interface SymptomReport {
  id: string;
  userId: string;
  createdAt: Date;
  symptoms: Symptom[];
}

export interface TriageResult {
  id: string;
  reportId: string;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  recommendation: string;
}
