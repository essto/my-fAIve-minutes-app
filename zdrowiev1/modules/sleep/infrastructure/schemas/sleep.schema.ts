import { pgTable, uuid, timestamp, integer, text } from 'drizzle-orm/pg-core';
import { users } from '../../../shared/database/src/drizzle/schema';

export const sleepRecords = pgTable('sleep_records', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  startTime: timestamp('start_time').notNull(),
  endTime: timestamp('end_time').notNull(),
  quality: integer('quality'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});
