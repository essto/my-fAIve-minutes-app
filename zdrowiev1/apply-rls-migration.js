const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

async function migrate() {
    const client = new Client({
        host: process.env.POSTGRES_HOST || 'localhost',
        port: parseInt(process.env.POSTGRES_PORT || '5432'),
        user: process.env.POSTGRES_USER || 'postgres',
        password: process.env.POSTGRES_PASSWORD || 'postgres',
        database: process.env.POSTGRES_DB || 'health',
    });

    await client.connect();
    console.log('Connected to database');

    const migrationPath = path.join(__dirname, 'modules/shared/database/src/migrations/0009_enable_rls_policies.sql');
    const sql = fs.readFileSync(migrationPath, 'utf8');

    try {
        await client.query(sql);
        console.log('Migration applied successfully');
    } catch (error) {
        console.error('Migration failed:', error);
    } finally {
        await client.end();
    }
}

migrate();
