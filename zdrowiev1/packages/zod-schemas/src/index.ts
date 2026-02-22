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

export const NutritionValueSchema = z.object({
  calories: z.number().int().min(0),
  protein: z.number().int().min(0),
  carbs: z.number().int().min(0),
  fat: z.number().int().min(0),
});

export const MealProductSchema = z.object({
  id: z.string().uuid().optional(),
  mealId: z.string().uuid().optional(),
  name: z.string().min(1),
  productId: z.string().optional(),
  barcode: z.string().optional(),
  quantity: z.number().int().min(1), // in grams
  calories: z.number().int().min(0),
  protein: z.number().int().min(0),
  carbs: z.number().int().min(0),
  fat: z.number().int().min(0),
});

export const MealSchema = z.object({
  id: z.string().uuid(),
  userId: z.string().uuid(),
  name: z.string().min(1),
  consumedAt: z.coerce.date().default(() => new Date()),
  products: z.array(MealProductSchema).optional(),
});

export const DailyNutritionSummarySchema = z.object({
  userId: z.string().uuid(),
  date: z.string(),
  total: NutritionValueSchema,
  target: NutritionValueSchema.optional(),
  isDeficit: z.boolean(),
  isSurplus: z.boolean(),
});

export const SymptomSchema = z.object({
  id: z.string().uuid().optional(),
  reportId: z.string().uuid().optional(),
  name: z.string().min(1),
  severity: z.number().int().min(1).max(10),
  durationHours: z.number().int().min(1),
});

export const SymptomReportSchema = z.object({
  id: z.string().uuid(),
  userId: z.string().uuid(),
  createdAt: z.coerce.date().default(() => new Date()),
  symptoms: z.array(SymptomSchema).optional(),
});

export const TriageResultSchema = z.object({
  id: z.string().uuid(),
  reportId: z.string().uuid(),
  riskLevel: z.enum(['LOW', 'MEDIUM', 'HIGH']),
  recommendation: z.string().min(1),
});

export const ActivityEntrySchema = z.object({
  id: z.string().uuid(),
  userId: z.string().uuid(),
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  steps: z.number().int().min(0).default(0),
  caloriesBurned: z.number().min(0).default(0),
  activityType: z.string().max(50).optional(),
  durationMinutes: z.number().int().min(0).optional(),
  createdAt: z.coerce.date().default(() => new Date()),
});
