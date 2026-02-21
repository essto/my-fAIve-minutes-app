import type { Config } from 'drizzle-kit';
import dotenv from 'dotenv';

dotenv.config();

export default {
    schema: './modules/shared/database/src/drizzle/schema.ts',
    out: './modules/shared/database/src/migrations',
    dialect: 'postgresql',
    dbCredentials: {
        host: process.env.POSTGRES_HOST || 'localhost',
        port: Number(process.env.POSTGRES_PORT) || 5432,
        user: process.env.POSTGRES_USER || 'postgres',
        password: process.env.POSTGRES_PASSWORD || 'postgres',
        database: process.env.POSTGRES_DB || 'health',
    },
} satisfies Config;
