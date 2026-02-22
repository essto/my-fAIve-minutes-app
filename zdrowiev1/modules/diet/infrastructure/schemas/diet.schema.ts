import { pgTable, uuid, timestamp, integer, text, doublePrecision } from 'drizzle-orm/pg-core';
import { users } from '@monorepo/database';

export const mealEntries = pgTable('meal_entries', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  name: text('name').notNull(),
  calories: integer('calories').notNull(),
  carbs: doublePrecision('carbs'),
  protein: doublePrecision('protein'),
  fat: doublePrecision('fat'),
  timestamp: timestamp('timestamp').defaultNow().notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});
