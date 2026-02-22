import {
  pgTable,
  uuid,
  integer,
  doublePrecision,
  timestamp,
  varchar,
  date,
} from 'drizzle-orm/pg-core';
import { users } from '../../../../shared/database/src/drizzle/schema';

export const activityEntries = pgTable('activity_entries', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  date: date('date').notNull(),
  steps: integer('steps').default(0).notNull(),
  caloriesBurned: doublePrecision('calories_burned').default(0).notNull(),
  activityType: varchar('activity_type', { length: 50 }),
  durationMinutes: integer('duration_minutes'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});
