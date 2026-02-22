import { db } from '@monorepo/database';
import {
  symptomReports,
  symptoms,
  triageResults,
} from '../../../../shared/database/src/drizzle/schema';
import { DiagnosisRepository } from '../../../domain/ports/diagnosis.repository';
import { SymptomReport, TriageResult } from '../../../domain/entities/symptom.entity';
import { eq } from 'drizzle-orm';

export class DrizzleDiagnosisRepository implements DiagnosisRepository {
  async saveReport(report: SymptomReport): Promise<SymptomReport> {
    return await db.transaction(async (tx) => {
      const [insertedReport] = await tx
        .insert(symptomReports)
        .values({
          id: report.id,
          userId: report.userId,
          createdAt: report.createdAt,
        })
        .onConflictDoUpdate({
          target: symptomReports.id,
          set: { createdAt: report.createdAt },
        })
        .returning();

      // Clear existing symptoms if updating
      await tx.delete(symptoms).where(eq(symptoms.reportId, report.id));

      if (report.symptoms.length > 0) {
        await tx.insert(symptoms).values(
          report.symptoms.map((s) => ({
            id: s.id || crypto.randomUUID(),
            reportId: report.id,
            name: s.name,
            severity: s.severity,
            durationHours: s.durationHours,
          })),
        );
      }

      return {
        ...insertedReport,
        symptoms: report.symptoms,
      };
    });
  }

  async saveTriageResult(result: TriageResult): Promise<TriageResult> {
    const [inserted] = await db
      .insert(triageResults)
      .values({
        id: result.id,
        reportId: result.reportId,
        riskLevel: result.riskLevel,
        recommendation: result.recommendation,
      })
      .onConflictDoUpdate({
        target: triageResults.id,
        set: { riskLevel: result.riskLevel, recommendation: result.recommendation },
      })
      .returning();

    return inserted as any;
  }

  async findReportsByUserId(userId: string): Promise<SymptomReport[]> {
    const reports = await db.select().from(symptomReports).where(eq(symptomReports.userId, userId));

    return await Promise.all(
      reports.map(async (report) => {
        const reportSymptoms = await db
          .select()
          .from(symptoms)
          .where(eq(symptoms.reportId, report.id));
        return {
          ...report,
          symptoms: reportSymptoms,
        };
      }),
    );
  }

  async findTriageResultByReportId(reportId: string): Promise<TriageResult | null> {
    const [result] = await db
      .select()
      .from(triageResults)
      .where(eq(triageResults.reportId, reportId));

    return (result as any) || null;
  }
}
