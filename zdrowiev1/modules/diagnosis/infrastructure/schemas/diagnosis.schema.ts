import { pgTable, uuid, timestamp, varchar, integer, text } from 'drizzle-orm/pg-core';
import { users } from '../../../shared/database/src/drizzle/schema';

export const symptomReports = pgTable('symptom_reports', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  description: text('description').notNull(),
  severity: integer('severity').notNull(),
  timestamp: timestamp('timestamp').defaultNow().notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const diagnoses = pgTable('diagnoses', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  symptomReportId: uuid('symptom_report_id')
    .notNull()
    .references(() => symptomReports.id, { onDelete: 'cascade' }),
  result: text('result').notNull(),
  confidence: doublePrecision('confidence').notNull(),
  recommendations: jsonb('recommendations').$type<string[]>().notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});
