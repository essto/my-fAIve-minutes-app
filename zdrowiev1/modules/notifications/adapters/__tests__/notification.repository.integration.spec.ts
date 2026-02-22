import { Pool } from 'pg';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { GenericContainer, StartedTestContainer } from 'testcontainers';
import { drizzle } from 'drizzle-orm/node-postgres';
import { DrizzleNotificationRepository } from '../drizzle-notification.repository';
import {
  Notification,
  NotificationType,
  NotificationChannel,
} from '../../domain/notification.entity';
import * as schema from '../../../shared/database/src/drizzle/schema';
import * as fs from 'fs';
import * as path from 'path';

describe('NotificationRepository Integration (RLS)', () => {
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

    // 1. Setup Auth Schema and Users Table
    await adminPool.query(`
      CREATE SCHEMA IF NOT EXISTS auth;
      CREATE TABLE auth.users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        first_name VARCHAR(255),
        last_name VARCHAR(255),
        is_demo BOOLEAN DEFAULT false,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
      );
      
      -- Mock the users table in public schema if needed by references
      CREATE TABLE public.users (LIKE auth.users INCLUDING ALL);
    `);

    // 2. Load and execute the notifications migration
    const migrationPath = path.resolve(
      __dirname,
      '../../../shared/database/src/migrations/0007_create_notifications.sql',
    );
    const migrationSql = fs.readFileSync(migrationPath, 'utf8');

    // Fix foreign key reference to point to auth.users if it's there, or public.users
    // In our migration it is: REFERENCES auth.users(id)
    await adminPool.query(migrationSql);

    // 3. Setup App Role and RLS Access
    await adminPool.query(`
      CREATE ROLE health_app_user WITH LOGIN PASSWORD 'app_password';
      GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO health_app_user;
      GRANT ALL PRIVILEGES ON SCHEMA auth TO health_app_user;
      GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA auth TO health_app_user;
    `);

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

  it('should isolate notifications between users via RLS', async () => {
    // 1. Create two users
    const userA = (
      await adminPool.query(
        `INSERT INTO auth.users (email, password) VALUES ('userA@test.com', 'hash') RETURNING id`,
      )
    ).rows[0].id;
    const userB = (
      await adminPool.query(
        `INSERT INTO auth.users (email, password) VALUES ('userB@test.com', 'hash') RETURNING id`,
      )
    ).rows[0].id;

    const db = drizzle(userPool, { schema });
    const repo = new DrizzleNotificationRepository(db as any);

    // 2. Insert notifications as admin (RLS bypassed)
    await adminPool.query(`
      INSERT INTO notifications (user_id, type, title, message, channel) 
      VALUES ('${userA}', 'SYSTEM', 'Welcome A', 'Hello User A', 'IN_APP');
      INSERT INTO notifications (user_id, type, title, message, channel) 
      VALUES ('${userB}', 'SYSTEM', 'Welcome B', 'Hello User B', 'IN_APP');
    `);

    // 3. Check as User A
    const clientA = await userPool.connect();
    try {
      await clientA.query(`SET app.current_user_id = '${userA}'`);
      const dbA = drizzle(clientA as any, { schema });
      const repoA = new DrizzleNotificationRepository(dbA as any);

      const notifsA = await repoA.findByUserId(userA);
      expect(notifsA).toHaveLength(1);
      expect(notifsA[0].title).toBe('Welcome A');

      // Try to get User B's notifications
      const notifsBForA = await repoA.findByUserId(userB);
      expect(notifsBForA).toHaveLength(0);
    } finally {
      clientA.release();
    }

    // 4. Check as User B
    const clientB = await userPool.connect();
    try {
      await clientB.query(`SET app.current_user_id = '${userB}'`);
      const dbB = drizzle(clientB as any, { schema });
      const repoB = new DrizzleNotificationRepository(dbB as any);

      const notifsB = await repoB.findByUserId(userB);
      expect(notifsB).toHaveLength(1);
      expect(notifsB[0].title).toBe('Welcome B');
    } finally {
      clientB.release();
    }
  });

  it('should create and find notification', async () => {
    const userId = (
      await adminPool.query(
        `INSERT INTO auth.users (email, password) VALUES ('test@test.com', 'hash') RETURNING id`,
      )
    ).rows[0].id;

    const client = await userPool.connect();
    try {
      await client.query(`SET app.current_user_id = '${userId}'`);
      const db = drizzle(client as any, { schema });
      const repo = new DrizzleNotificationRepository(db as any);

      const notif = Notification.create({
        userId,
        type: NotificationType.HEALTH_ALERT,
        title: 'Anomaly',
        message: 'High HR',
        channel: NotificationChannel.IN_APP,
      });

      const saved = await repo.create(notif);
      expect(saved.id).toBeDefined();

      const found = await repo.findById(saved.id!);
      expect(found).toBeDefined();
      expect(found!.title).toBe('Anomaly');
    } finally {
      client.release();
    }
  });
});
