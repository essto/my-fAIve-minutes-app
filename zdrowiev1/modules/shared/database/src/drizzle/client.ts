import { drizzle } from 'drizzle-orm/node-postgres';
import { Pool } from 'pg';
import * as schema from './schema';
import dotenv from 'dotenv';
import path from 'path';

// Load .env from root
dotenv.config({ path: path.resolve(__dirname, '../../../../../../.env') });

const pool = new Pool({
    host: process.env.POSTGRES_HOST || 'localhost',
    port: Number(process.env.POSTGRES_PORT) || 5432,
    user: process.env.POSTGRES_USER || 'postgres',
    password: process.env.POSTGRES_PASSWORD || 'postgres',
    database: process.env.POSTGRES_DB || 'health',
});

export const db = drizzle(pool, { schema });
