import { z } from 'zod';

export const GetWeightHistoryInputSchema = z
  .object({
    user_id: z.string().uuid().describe('ID użytkownika'),
    limit: z.number().int().min(1).max(100).default(20).describe('Maks. wyników'),
    offset: z.number().int().min(0).default(0).describe('Paginacja offset'),
  })
  .strict();

export const AddWeightReadingInputSchema = z
  .object({
    user_id: z.string().uuid().describe('ID użytkownika'),
    value: z.number().min(0.1).max(500).describe('Wartość wagi w podanej jednostce (0.1 - 500)'),
    unit: z.enum(['kg', 'lbs']).default('kg').describe('Jednostka wagi'),
  })
  .strict();

export const GetHealthScoreInputSchema = z
  .object({
    user_id: z.string().uuid().describe('ID użytkownika'),
  })
  .strict();

export type GetWeightHistoryInput = z.infer<typeof GetWeightHistoryInputSchema>;
export type AddWeightReadingInput = z.infer<typeof AddWeightReadingInputSchema>;
export type GetHealthScoreInput = z.infer<typeof GetHealthScoreInputSchema>;
