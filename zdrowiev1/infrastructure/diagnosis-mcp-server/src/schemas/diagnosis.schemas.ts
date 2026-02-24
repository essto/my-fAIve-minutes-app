import { z } from 'zod';

export const SubmitSymptomsInputSchema = z
  .object({
    user_id: z.string().uuid().describe('ID użytkownika'),
    symptoms: z.array(z.string()).min(1).describe("Lista objawów (np. 'ból głowy', 'gorączka')"),
  })
  .strict();

export const GetTriageInputSchema = z
  .object({
    report_id: z.string().uuid().describe('ID raportu z objawami'),
  })
  .strict();

export const GenerateReportInputSchema = z
  .object({
    report_id: z.string().uuid().describe('ID raportu do wygenerowania PDF'),
    format: z.enum(['json', 'pdf']).default('json').describe('Format wyjściowy'),
  })
  .strict();

export type SubmitSymptomsInput = z.infer<typeof SubmitSymptomsInputSchema>;
export type GetTriageInput = z.infer<typeof GetTriageInputSchema>;
export type GenerateReportInput = z.infer<typeof GenerateReportInputSchema>;
