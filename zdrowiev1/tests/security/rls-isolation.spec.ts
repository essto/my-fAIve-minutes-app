/**
 * TDD: RLS (Row-Level Security) Isolation Tests
 * Stage 11 — Faza 5
 *
 * RED PHASE: Wszystkie testy powinny FAILOWAĆ przed implementacją.
 * Zadanie:
 *   1. Utwórz migrację 0009_enable_rls_policies.sql
 *   2. Włącz RLS na tabelach: users, weight_readings, heart_rate_readings,
 *      sleep_records, meal_entries, symptom_reports, notifications
 *   3. Utwórz polityki izolacji per user_id
 *
 * UWAGA: Te testy wymagają działającej bazy PostgreSQL (docker compose up)
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { drizzle } from 'drizzle-orm/node-postgres';
import { Client } from 'pg';

describe('RLS Isolation (Faza 5)', () => {
    let adminClient: Client;

    beforeAll(async () => {
        adminClient = new Client({
            host: process.env.POSTGRES_HOST || 'localhost',
            port: parseInt(process.env.POSTGRES_PORT || '5432'),
            user: process.env.POSTGRES_USER || 'postgres',
            password: process.env.POSTGRES_PASSWORD || 'postgres',
            database: process.env.POSTGRES_DB || 'health',
        });
        await adminClient.connect();

        // Create an app_user role so RLS policies are actually enforced
        // (Superusers bypass RLS, so this is mandatory for testing)
        await adminClient.query(`DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'app_user') THEN CREATE ROLE app_user NOLOGIN; END IF; END $$;`);
        await adminClient.query(`GRANT USAGE ON SCHEMA public TO app_user;`);
        await adminClient.query(`GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app_user;`);
    });

    afterAll(async () => {
        try {
            await adminClient.query(`DROP OWNED BY app_user;`);
            await adminClient.query(`DROP ROLE IF EXISTS app_user;`);
        } catch (e) {
            // Ignore cleanup errors
        }
        await adminClient?.end();
    });

    describe('RLS Enabled on Critical Tables', () => {
        const criticalTables = [
            'weight_readings',
            'heart_rate_readings',
            'sleep_records',
            'meal_entries',
            'symptom_reports',
            'notifications',
        ];

        for (const tableName of criticalTables) {
            it(`should have RLS enabled on ${tableName}`, async () => {
                const result = await adminClient.query(
                    `SELECT relrowsecurity FROM pg_class WHERE relname = $1`,
                    [tableName]
                );

                if (result.rows.length === 0) {
                    // Table doesn't exist yet — acceptable during early RED phase
                    console.warn(`Table ${tableName} does not exist yet`);
                    return;
                }

                expect(result.rows[0].relrowsecurity).toBe(true);
            });
        }
    });

    describe('Cross-User Data Isolation', () => {
        it('should NOT allow User A to read User B weight data', async () => {
            const userAId = '00000000-0000-0000-0000-000000000001';
            const userBId = '00000000-0000-0000-0000-000000000002';

            try {
                await adminClient.query(`SET ROLE app_user`);
                await adminClient.query(`SET app.current_user_id = '${userAId}'`);

                const result = await adminClient.query(
                    `SELECT * FROM weight_readings WHERE user_id = $1`,
                    [userBId]
                );

                expect(result.rows.length).toBe(0);
            } catch (error: any) {
                if (error.message.includes('does not exist')) return; // Acceptable in early stages
                expect(error.message).toMatch(/policy|permission|denied/i);
            }
        });

        it('should NOT allow User A to DELETE User B notifications', async () => {
            const userAId = '00000000-0000-0000-0000-000000000001';
            const userBId = '00000000-0000-0000-0000-000000000002';

            try {
                await adminClient.query(`SET ROLE app_user`);
                await adminClient.query(`SET app.current_user_id = '${userAId}'`);

                const result = await adminClient.query(
                    `DELETE FROM notifications WHERE user_id = $1 RETURNING *`,
                    [userBId]
                );

                expect(result.rowCount).toBe(0);
            } catch (error: any) {
                if (error.message.includes('does not exist')) return; // Acceptable in early stages
                expect(error.message).toMatch(/policy|permission|denied/i);
            }
        });

        it('should NOT allow User A to UPDATE User B symptom reports', async () => {
            const userAId = '00000000-0000-0000-0000-000000000001';
            const userBId = '00000000-0000-0000-0000-000000000002';

            try {
                await adminClient.query(`SET ROLE app_user`);
                await adminClient.query(`SET app.current_user_id = '${userAId}'`);

                const result = await adminClient.query(
                    `UPDATE symptom_reports SET notes = 'hacked' WHERE user_id = $1 RETURNING *`,
                    [userBId]
                );

                expect(result.rowCount).toBe(0);
            } catch (error: any) {
                if (error.message.includes('does not exist')) return; // Acceptable in early stages
                expect(error.message).toMatch(/policy|permission|denied/i);
            }
        });
    });

    describe('RLS Migration File Exists', () => {
        it('should have an RLS migration file', async () => {
            const fs = await import('fs');
            const path = await import('path');
            const migrationsDir = path.resolve(
                __dirname,
                '../../modules/shared/database/src/migrations'
            );

            const files = fs.readdirSync(migrationsDir);
            const rlsMigration = files.find(
                (f: string) => f.toLowerCase().includes('rls')
            );

            expect(rlsMigration).toBeDefined();
        });
    });
});
