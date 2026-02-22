import {
  pgTable,
  uuid,
  text,
  timestamp,
  doublePrecision,
  varchar,
  boolean,
  integer,
} from 'drizzle-orm/pg-core';

export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: text('email').notNull().unique(),
  password: text('password').notNull(),
  firstName: varchar('first_name', { length: 255 }),
  lastName: varchar('last_name', { length: 255 }),
  isDemo: boolean('is_demo').default(false).notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

export const consents = pgTable('consents', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  category: varchar('category', { length: 50 }).notNull(), // 'weight', 'heart_rate', etc.
  status: varchar('status', { length: 20 }).notNull(), // 'granted', 'revoked'
  grantedAt: timestamp('granted_at').defaultNow().notNull(),
  revokedAt: timestamp('revoked_at'),
});

export const auditLogs = pgTable('audit_logs', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id),
  action: text('action').notNull(),
  metadata: text('metadata'),
  timestamp: timestamp('timestamp').defaultNow().notNull(),
});

export const notifications = pgTable('notifications', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  type: varchar('type', { length: 20 }).notNull(), // 'SYSTEM', 'HEALTH_ALERT', 'REMINDER'
  title: varchar('title', { length: 100 }).notNull(),
  message: text('message').notNull(),
  channel: varchar('channel', { length: 20 }).notNull(), // 'IN_APP', 'EMAIL', 'PUSH'
  isRead: boolean('is_read').notNull().default(false),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  readAt: timestamp('read_at'),
});

export const notificationPreferences = pgTable(
  'notification_preferences',
  {
    userId: uuid('user_id')
      .notNull()
      .references(() => users.id, { onDelete: 'cascade' }),
    type: varchar('type', { length: 20 }).notNull(),
    channel: varchar('channel', { length: 20 }).notNull(),
    enabled: boolean('enabled').notNull().default(true),
  },
  (table) => {
    return {
      pk: [table.userId, table.type, table.channel],
    };
  },
);

// --- DIET MODULE ---

export const meals = pgTable('meals', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  name: varchar('name', { length: 255 }).notNull(),
  consumedAt: timestamp('consumed_at').defaultNow().notNull(),
});

export const mealProducts = pgTable('meal_products', {
  id: uuid('id').primaryKey().defaultRandom(),
  mealId: uuid('meal_id')
    .notNull()
    .references(() => meals.id, { onDelete: 'cascade' }),
  name: varchar('name', { length: 255 }).notNull(),
  barcode: varchar('barcode', { length: 50 }),
  productId: varchar('product_id', { length: 255 }),
  quantity: integer('quantity').notNull(), // grams
  calories: integer('calories').notNull(),
  protein: integer('protein').notNull(),
  carbs: integer('carbs').notNull(),
  fat: integer('fat').notNull(),
});

// --- DIAGNOSIS MODULE ---

export const symptomReports = pgTable('symptom_reports', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export const symptoms = pgTable('symptoms', {
  id: uuid('id').primaryKey().defaultRandom(),
  reportId: uuid('report_id')
    .notNull()
    .references(() => symptomReports.id, { onDelete: 'cascade' }),
  name: varchar('name', { length: 100 }).notNull(),
  severity: integer('severity').notNull(), // 1-10
  durationHours: integer('duration_hours').notNull(),
});

export const triageResults = pgTable('triage_results', {
  id: uuid('id').primaryKey().defaultRandom(),
  reportId: uuid('report_id')
    .notNull()
    .references(() => symptomReports.id, { onDelete: 'cascade' }),
  riskLevel: varchar('risk_level', { length: 20 }), // LOW, MEDIUM, HIGH
  recommendation: text('recommendation').notNull(),
});
