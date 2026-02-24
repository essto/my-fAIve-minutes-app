import { pgTable, uuid, doublePrecision, timestamp, text } from 'drizzle-orm/pg-core';
import { users } from '../../../shared/database/src/drizzle/schema';

export const weightReadings = pgTable('weight_readings', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  value: doublePrecision('value').notNull(),
  unit: text('unit').default('kg').notNull(),
  bmi: doublePrecision('bmi'),
  fatPercent: doublePrecision('fat_percent'),
  fatKg: doublePrecision('fat_kg'),
  muscleMassKg: doublePrecision('muscle_mass_kg'),
  musclePercent: doublePrecision('muscle_percent'),
  waterPercent: doublePrecision('water_percent'),
  bmrKcal: doublePrecision('bmr_kcal'),
  boneMassKg: doublePrecision('bone_mass_kg'),
  proteinPercent: doublePrecision('protein_percent'),
  leanMassKg: doublePrecision('lean_mass_kg'),
  metabolicAge: doublePrecision('metabolic_age'),
  timestamp: timestamp('timestamp').defaultNow().notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});
