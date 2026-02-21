import { db } from '../../../../shared/database/src/index';
import { symptomReports, diagnoses } from '../../schemas/diagnosis.schema';
import { DiagnosisRepository } from '../../../domain/ports/diagnosis.repository';
import { SymptomReport, Diagnosis } from '@monorepo/shared-types';
import { eq } from 'drizzle-orm';

export class DrizzleDiagnosisRepository implements DiagnosisRepository {
  async saveSymptomReport(data: Partial<SymptomReport>): Promise<SymptomReport> {
    const [inserted] = await db
      .insert(symptomReports)
      .values(data as any)
      .returning();
    return inserted as unknown as SymptomReport;
  }

  async saveDiagnosis(data: Partial<Diagnosis>): Promise<Diagnosis> {
    const [inserted] = await db
      .insert(diagnoses)
      .values(data as any)
      .returning();
    return inserted as unknown as Diagnosis;
  }

  async findReportsByUserId(userId: string): Promise<SymptomReport[]> {
    const results = await db.select().from(symptomReports).where(eq(symptomReports.userId, userId));
    return results as unknown as SymptomReport[];
  }

  async findDiagnosesByUserId(userId: string): Promise<Diagnosis[]> {
    const results = await db.select().from(diagnoses).where(eq(diagnoses.userId, userId));
    return results as unknown as Diagnosis[];
  }
}
