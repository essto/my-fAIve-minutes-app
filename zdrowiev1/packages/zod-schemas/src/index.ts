import { z } from 'zod';

export const UserSchema = z.object({
    id: z.string().uuid(),
    email: z.string().email(),
    firstName: z.string().min(1).optional(),
    lastName: z.string().min(1).optional(),
    age: z.number().int().min(0).max(120).optional(),
    height: z.number().min(0).max(300).optional(),
    gender: z.enum(['male', 'female', 'other']).optional(),
    createdAt: z.date().default(() => new Date()),
    updatedAt: z.date().default(() => new Date()),
});

export const ConsentSchema = z.object({
    id: z.string().uuid(),
    userId: z.string().uuid(),
    category: z.enum(['weight', 'heart_rate', 'sleep', 'diet', 'diagnosis']),
    status: z.enum(['granted', 'revoked']),
    grantedAt: z.date().default(() => new Date()),
    revokedAt: z.date().optional(),
});

export const WeightReadingSchema = z.object({
    id: z.string().uuid(),
    userId: z.string().uuid(),
    value: z.number().min(0.5).max(500),
    unit: z.enum(['kg', 'lbs']).default('kg'),
    timestamp: z.date().default(() => new Date()),
    source: z.string().optional(),
});
