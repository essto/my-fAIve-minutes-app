import { pgTable, uuid, integer, timestamp, boolean } from 'drizzle-orm/pg-core';
import { users } from '../../../shared/database/src/drizzle/schema';

export const heartRateReadings = pgTable('heart_rate_readings', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  value: integer('value').notNull(),
  isResting: boolean('is_resting').default(false).notNull(),
  timestamp: timestamp('timestamp').defaultNow().notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});
