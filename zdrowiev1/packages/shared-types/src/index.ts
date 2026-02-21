import { z } from 'zod';
import * as schemas from '@monorepo/zod-schemas';

export type User = z.infer<typeof schemas.UserSchema>;
export type Consent = z.infer<typeof schemas.ConsentSchema>;
export type WeightReading = z.infer<typeof schemas.WeightReadingSchema>;
export type HeartRateReading = z.infer<typeof schemas.HeartRateReadingSchema>;
export type SleepRecord = z.infer<typeof schemas.SleepRecordSchema>;
export type MealEntry = z.infer<typeof schemas.MealEntrySchema>;
export type SymptomReport = z.infer<typeof schemas.SymptomReportSchema>;
export type Diagnosis = z.infer<typeof schemas.DiagnosisSchema>;
