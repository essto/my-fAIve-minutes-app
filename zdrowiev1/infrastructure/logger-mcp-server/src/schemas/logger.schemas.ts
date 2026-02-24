import { z } from 'zod';

export const GetRecentErrorsInputSchema = z
  .object({
    limit: z.number().int().min(1).max(100).default(20),
    level: z.enum(['ERROR', 'WARN', 'INFO']).optional(),
  })
  .strict();

export const GetErrorPatternsInputSchema = z
  .object({
    timeframe_hours: z.number().int().min(1).max(72).default(24),
  })
  .strict();

export const SuggestFixInputSchema = z
  .object({
    error_pattern: z.string().min(1),
  })
  .strict();

export type GetRecentErrorsInput = z.infer<typeof GetRecentErrorsInputSchema>;
export type GetErrorPatternsInput = z.infer<typeof GetErrorPatternsInputSchema>;
export type SuggestFixInput = z.infer<typeof SuggestFixInputSchema>;
