import { z } from 'zod';
import { ActivityEntrySchema } from '@monorepo/zod-schemas';

export const CreateActivityCommandSchema = ActivityEntrySchema.omit({
  id: true,
  userId: true,
  createdAt: true,
});

export type CreateActivityCommand = z.infer<typeof CreateActivityCommandSchema>;

export const ActivityDTOSchema = ActivityEntrySchema;
export type ActivityDTO = z.infer<typeof ActivityDTOSchema>;
