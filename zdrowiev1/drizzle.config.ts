import type { Config } from 'drizzle-kit';
import dotenv from 'dotenv';

dotenv.config();

const databaseUrl = `postgresql://${process.env.POSTGRES_USER || 'postgres'}:${process.env.POSTGRES_PASSWORD || 'postgres'}@${process.env.POSTGRES_HOST || 'localhost'}:${process.env.POSTGRES_PORT || 5432}/${process.env.POSTGRES_DB || 'health'}`;

export default {
  schema: [
    './modules/shared/database/src/drizzle/schema.ts',
    './modules/weight/infrastructure/schemas/weight.schema.ts',
    './modules/heart-rate/infrastructure/schemas/heart-rate.schema.ts',
    './modules/sleep/infrastructure/schemas/sleep.schema.ts',
    './modules/diet/infrastructure/schemas/diet.schema.ts',
    './modules/diagnosis/infrastructure/schemas/diagnosis.schema.ts',
    './modules/activity/src/infrastructure/schemas/activity.schema.ts',
  ],
  out: './modules/shared/database/src/migrations',
  driver: 'pg',
  dialect: 'postgresql',
  dbCredentials: {
    connectionString: databaseUrl,
  },
};
