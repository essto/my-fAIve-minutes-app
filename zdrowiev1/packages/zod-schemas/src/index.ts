import { z } from 'zod';

export const UserSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  password: z.string().min(8),
  firstName: z.string().min(1).optional(),
  lastName: z.string().min(1).optional(),
  age: z.number().int().min(0).max(120).optional(),
  height: z.number().min(0).max(300).optional(),
  gender: z.enum(['male', 'female', 'other']).optional(),
  isDemo: z.boolean().default(false),
  createdAt: z.coerce.date().default(() => new Date()),
  updatedAt: z.coerce.date().default(() => new Date()),
});

export const ConsentSchema = z.object({
  id: z.string().uuid(),
  userId: z.string().uuid(),
  category: z.enum(['weight', 'heart_rate', 'sleep', 'diet', 'diagnosis']),
  status: z.enum(['granted', 'revoked']),
  grantedAt: z.coerce.date().default(() => new Date()),
  revokedAt: z.coerce.date().optional(),
});

export const WeightReadingSchema = z.object({
  id: z.string().uuid(),
  userId: z.string().uuid(),
  value: z.number().min(0.5).max(500),
  unit: z.enum(['kg', 'lbs']).default('kg'),
  timestamp: z.coerce.date().default(() => new Date()),
  source: z.string().optional(),
});

export const HeartRateReadingSchema = z.object({
  id: z.string().uuid(),
  userId: z.string().uuid(),
  value: z.number().int().min(30).max(250),
  isResting: z.boolean().default(false),
  timestamp: z.coerce.date().default(() => new Date()),
});

export const SleepRecordSchema = z.object({
  id: z.string().uuid(),
  userId: z.string().uuid(),
  startTime: z.coerce.date(),
  endTime: z.coerce.date(),
  quality: z.number().int().min(1).max(10).optional(),
  createdAt: z.coerce.date().default(() => new Date()),
});

export const MealEntrySchema = z.object({
  id: z.string().uuid(),
  userId: z.string().uuid(),
  name: z.string().min(1),
  calories: z.number().int().min(0),
  carbs: z.number().optional(),
  protein: z.number().optional(),
  fat: z.number().optional(),
  timestamp: z.coerce.date().default(() => new Date()),
});

export const SymptomReportSchema = z.object({
  id: z.string().uuid(),
  userId: z.string().uuid(),
  description: z.string().min(1),
  severity: z.number().int().min(1).max(10),
  timestamp: z.coerce.date().default(() => new Date()),
});

export const DiagnosisSchema = z.object({
  id: z.string().uuid(),
  userId: z.string().uuid(),
  symptomReportId: z.string().uuid(),
  result: z.string().min(1),
  confidence: z.number().min(0).max(1),
  recommendations: z.array(z.string()),
  createdAt: z.coerce.date().default(() => new Date()),
});
