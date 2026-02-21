import { z } from 'zod';
import { UserSchema, ConsentSchema, WeightReadingSchema } from '@monorepo/zod-schemas';

export type User = z.infer<typeof UserSchema>;
export type Consent = z.infer<typeof ConsentSchema>;
export type WeightReading = z.infer<typeof WeightReadingSchema>;

export type Category = Consent['category'];
export type ConsentStatus = Consent['status'];
