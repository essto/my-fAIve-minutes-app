import { drizzle } from 'drizzle-orm/node-postgres';
import { Pool } from 'pg';
import { afterAll, beforeAll, describe, expect, it, beforeEach } from 'vitest';
import { GenericContainer, StartedTestContainer } from 'testcontainers';
import { DrizzleWeightRepository } from '../infrastructure/database/repository/drizzle-weight.repository';
import { weightReadings } from '../infrastructure/schemas/weight.schema';
import { users } from '@monorepo/database';
import { sql } from 'drizzle-orm';

describe('WeightRepository (Drizzle) Contract Test', () => {
  let container: StartedTestContainer;
  let pool: Pool;
  let db: any;
  let repository: DrizzleWeightRepository;
  let testUserId: string;

  beforeAll(async () => {
    container = await new GenericContainer('postgres:15-alpine')
      .withExposedPorts(5432)
      .withEnvironment({
        POSTGRES_USER: 'test',
        POSTGRES_PASSWORD: 'test',
        POSTGRES_DB: 'testdb',
      })
      .start();

    const host = container.getHost();
    const port = container.getMappedPort(5432);

    pool = new Pool({
      host,
      port,
      user: 'test',
      password: 'test',
      database: 'testdb',
    });
    db = drizzle(pool);
    repository = new DrizzleWeightRepository(db);

    // Setup schema
    await pool.query(`
            CREATE TABLE users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            );

            CREATE TABLE weight_readings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                value DOUBLE PRECISION NOT NULL,
                unit VARCHAR(10) DEFAULT 'kg',
                bmi DOUBLE PRECISION,
                fat_percent DOUBLE PRECISION,
                fat_kg DOUBLE PRECISION,
                muscle_mass_kg DOUBLE PRECISION,
                muscle_percent DOUBLE PRECISION,
                water_percent DOUBLE PRECISION,
                bmr_kcal DOUBLE PRECISION,
                bone_mass_kg DOUBLE PRECISION,
                protein_percent DOUBLE PRECISION,
                lean_mass_kg DOUBLE PRECISION,
                metabolic_age DOUBLE PRECISION,
                source VARCHAR(50),
                timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
        `);

    // Create a test user
    const res = await pool.query(`INSERT INTO users (email) VALUES ('test@repo.com') RETURNING id`);
    testUserId = res.rows[0].id;
  }, 60000);

  afterAll(async () => {
    if (pool) await pool.end();
    if (container) await container.stop();
  });

  beforeEach(async () => {
    await pool.query('DELETE FROM weight_readings');
  });

  it('should correctly save and find a weight reading', async () => {
    const reading = {
      userId: testUserId,
      value: 75.5,
      timestamp: new Date('2023-01-01T12:00:00Z'),
    };

    const saved = await repository.save(reading);
    expect(saved.id).toBeDefined();
    expect(saved.value).toBe(75.5);

    const history = await repository.findByUserId(testUserId);
    expect(history).toHaveLength(1);
    expect(history[0].value).toBe(75.5);
  });

  it('should reject invalid weight entries at DB level (null user_id)', async () => {
    const invalidReading = {
      value: 70,
    } as any;

    await expect(repository.save(invalidReading)).rejects.toThrow();
  });

  it('should correctly map decimal values', async () => {
    const reading = {
      userId: testUserId,
      value: 82.345,
      timestamp: new Date(),
    };

    const saved = await repository.save(reading);
    const fetched = await repository.findByUserId(testUserId);
    expect(fetched[0].value).toBe(82.345);
  });

  it('saves and retrieves bmi and fatPercent', async () => {
    const saved = await repository.save({
      userId: testUserId,
      value: 83.0,
      unit: 'kg',
      bmi: 26.3,
      fatPercent: 22.1,
      muscleMassKg: 38.5,
      waterPercent: 56.0,
      metabolicAge: 40,
      timestamp: new Date('2024-06-01'),
    });

    expect(saved.bmi).toBe(26.3);
    expect(saved.fatPercent).toBe(22.1);

    const found = await repository.findByUserId(testUserId);
    expect(found[0].bmi).toBeCloseTo(26.3);
  });
});
