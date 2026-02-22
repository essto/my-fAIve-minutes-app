import { pgTable, uuid, doublePrecision, timestamp, text } from 'drizzle-orm/pg-core';
import { users } from '../../../shared/database/src/drizzle/schema';

export const weightReadings = pgTable('weight_readings', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  value: doublePrecision('value').notNull(),
  timestamp: timestamp('timestamp').defaultNow().notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});
