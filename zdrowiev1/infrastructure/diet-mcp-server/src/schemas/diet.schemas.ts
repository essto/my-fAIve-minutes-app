import { z } from 'zod';

export const LogMealInputSchema = z
  .object({
    user_id: z.string().uuid().describe('ID użytkownika'),
    name: z.string().min(1).max(255).describe('Nazwa posiłku (np. Owsianka)'),
    calories: z.number().int().min(0).describe('Liczba kalorii'),
    protein: z.number().min(0).optional().describe('Białko w gramach'),
    carbs: z.number().min(0).optional().describe('Węglowodany w gramach'),
    fat: z.number().min(0).optional().describe('Tłuszcze w gramach'),
  })
  .strict();

export const GetDailySummaryInputSchema = z
  .object({
    user_id: z.string().uuid().describe('ID użytkownika'),
    date: z
      .string()
      .regex(/^\d{4}-\d{2}-\d{2}$/, 'Format daty RRRR-MM-DD')
      .describe('Data do podsumowania'),
  })
  .strict();

export const SearchFoodInputSchema = z
  .object({
    query: z.string().min(1).describe('Wyszukiwana fraza (np. jabłko)'),
  })
  .strict();

export type LogMealInput = z.infer<typeof LogMealInputSchema>;
export type GetDailySummaryInput = z.infer<typeof GetDailySummaryInputSchema>;
export type SearchFoodInput = z.infer<typeof SearchFoodInputSchema>;
