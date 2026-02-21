import { Pool } from 'pg';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { GenericContainer, StartedTestContainer } from 'testcontainers';

describe('Row Level Security (RLS) Integration Tests', () => {
  let container: StartedTestContainer;
  let adminPool: Pool;
  let userPool: Pool;

  beforeAll(async () => {
    container = await new GenericContainer('postgres:15-alpine')
      .withExposedPorts(5432)
      .withEnvironment({
        POSTGRES_USER: 'postgres',
        POSTGRES_PASSWORD: 'password',
        POSTGRES_DB: 'healthdb',
      })
      .start();

    const host = container.getHost();
    const port = container.getMappedPort(5432);

    adminPool = new Pool({
      host,
      port,
      user: 'postgres',
      password: 'password',
      database: 'healthdb',
    });

    // 1. Setup Database, Role and RLS as superuser (postgres)
    await adminPool.query(`
            CREATE ROLE health_app_user WITH LOGIN PASSWORD 'app_password';
            
            CREATE TABLE users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );

            CREATE TABLE weight_readings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                value DOUBLE PRECISION NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW()
            );

            GRANT ALL PRIVILEGES ON TABLE users TO health_app_user;
            GRANT ALL PRIVILEGES ON TABLE weight_readings TO health_app_user;

            ALTER TABLE weight_readings ENABLE ROW LEVEL SECURITY;
            -- No FORCE needed if we test with non-owner, but good for safety.
            
            CREATE POLICY user_isolation_policy ON weight_readings
            FOR ALL
            TO health_app_user
            USING (user_id = current_setting('app.current_user_id')::UUID);
        `);

    // 2. Setup a pool for the non-superuser
    userPool = new Pool({
      host,
      port,
      user: 'health_app_user',
      password: 'app_password',
      database: 'healthdb',
    });
  }, 60000);

  afterAll(async () => {
    if (adminPool) await adminPool.end();
    if (userPool) await userPool.end();
    if (container) await container.stop();
  });

  it('should isolate data between users for a non-superuser role', async () => {
    // Setup data as admin (RLS irrelevant)
    const userAId = (
      await adminPool.query(`INSERT INTO users (email) VALUES ('userA@example.com') RETURNING id`)
    ).rows[0].id;
    const userBId = (
      await adminPool.query(`INSERT INTO users (email) VALUES ('userB@example.com') RETURNING id`)
    ).rows[0].id;

    await adminPool.query(`INSERT INTO weight_readings (user_id, value) VALUES ('${userAId}', 70)`);
    await adminPool.query(`INSERT INTO weight_readings (user_id, value) VALUES ('${userBId}', 80)`);

    const client = await userPool.connect();
    try {
      // Verify User A isolation
      await client.query(`SET app.current_user_id = '${userAId}'`);
      const userAVisible = (await client.query('SELECT * FROM weight_readings')).rows;
      expect(userAVisible).toHaveLength(1);
      expect(userAVisible[0].user_id).toBe(userAId);
      expect(userAVisible[0].value).toBe(70);

      // Verify User B isolation
      await client.query(`SET app.current_user_id = '${userBId}'`);
      const userBVisible = (await client.query('SELECT * FROM weight_readings')).rows;
      expect(userBVisible).toHaveLength(1);
      expect(userBVisible[0].user_id).toBe(userBId);
      expect(userBVisible[0].value).toBe(80);
    } finally {
      client.release();
    }
  });

  it('should block unauthorized updates via RLS for a non-superuser role', async () => {
    const userAId = (
      await adminPool.query(
        `INSERT INTO users (email) VALUES ('userA_mod@example.com') RETURNING id`,
      )
    ).rows[0].id;
    const userBId = (
      await adminPool.query(
        `INSERT INTO users (email) VALUES ('userB_mod@example.com') RETURNING id`,
      )
    ).rows[0].id;
    const recordAId = (
      await adminPool.query(
        `INSERT INTO weight_readings (user_id, value) VALUES ('${userAId}', 75) RETURNING id`,
      )
    ).rows[0].id;

    const client = await userPool.connect();
    try {
      // User B tries to update User A's record
      await client.query(`SET app.current_user_id = '${userBId}'`);
      const updateResult = await client.query(
        'UPDATE weight_readings SET value = 99 WHERE id = $1',
        [recordAId],
      );
      expect(updateResult.rowCount).toBe(0);

      // Verify it didn't change (as User A)
      await client.query(`SET app.current_user_id = '${userAId}'`);
      const recordA = (
        await client.query('SELECT * FROM weight_readings WHERE id = $1', [recordAId])
      ).rows;
      expect(recordA[0].value).toBe(75);
    } finally {
      client.release();
    }
  });
});
