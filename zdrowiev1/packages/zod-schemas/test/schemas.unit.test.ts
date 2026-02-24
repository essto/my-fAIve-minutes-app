import { describe, it, expect } from 'vitest';
import { UserSchema, ConsentSchema, WeightReadingSchema } from '../src/index';

describe('Zod Schemas', () => {
  describe('UserSchema', () => {
    it('validates a valid user', () => {
      const validUser = {
        id: '550e8400-e29b-41d4-a716-446655440000',
        email: 'test@example.com',
        password: 'StrongPass123!',
      };
      expect(() => UserSchema.parse(validUser)).not.toThrow();
    });

    it('throws error for invalid email', () => {
      const invalidUser = {
        id: '550e8400-e29b-41d4-a716-446655440000',
        email: 'invalid-email',
      };
      expect(() => UserSchema.parse(invalidUser)).toThrow();
    });
  });

  describe('ConsentSchema', () => {
    it('validates a valid consent', () => {
      const validConsent = {
        id: '550e8400-e29b-41d4-a716-446655440000',
        userId: '550e8400-e29b-41d4-a716-446655440000',
        category: 'weight',
        status: 'granted',
      };
      expect(() => ConsentSchema.parse(validConsent)).not.toThrow();
    });
  });

  describe('WeightReadingSchema', () => {
    it('validates a valid weight reading', () => {
      const validReading = {
        id: '550e8400-e29b-41d4-a716-446655440000',
        userId: '550e8400-e29b-41d4-a716-446655440000',
        value: 75.5,
      };
      expect(() => WeightReadingSchema.parse(validReading)).not.toThrow();
    });

    it('rejects weight value out of range', () => {
      const invalidReading = {
        id: '550e8400-e29b-41d4-a716-446655440000',
        userId: '550e8400-e29b-41d4-a716-446655440000',
        value: 600,
      };
      expect(() => WeightReadingSchema.parse(invalidReading)).toThrow();
    });
  });
});
