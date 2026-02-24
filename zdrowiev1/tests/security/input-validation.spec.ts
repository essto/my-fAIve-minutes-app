/**
 * TDD: Input Validation DTO Tests
 * Stage 11 — Faza 3
 *
 * RED PHASE: Wszystkie testy powinny FAILOWAĆ przed implementacją.
 * Zadanie:
 *   1. Utwórz modules/shared/auth/src/dto/auth.dto.ts z RegisterDto i LoginDto
 *   2. Zamień @Body() any na @Body() RegisterDto / LoginDto w auth.controller.ts
 *   3. Upewnij się, że ValidationPipe jest aktywny globalnie
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import { NestExpressApplication } from '@nestjs/platform-express';
import { APP_GUARD } from '@nestjs/core';
import { AppModule } from '../../apps/api/src/app.module';

describe('Input Validation DTOs (Faza 3)', () => {
    let app: INestApplication;
    const baseUrl = 'http://localhost:3097';

    beforeAll(async () => {
        const moduleFixture: TestingModule = await Test.createTestingModule({
            imports: [AppModule],
        })
            .compile();

        app = moduleFixture.createNestApplication<NestExpressApplication>();
        (app as any).set('trust proxy', 1);
        app.setGlobalPrefix('api');
        app.useGlobalPipes(new ValidationPipe({
            whitelist: true,
            forbidNonWhitelisted: true,
            transform: true,
        }));
        await app.listen(3097);
    });

    afterAll(async () => {
        await app?.close();
    });

    describe('Register DTO Validation', () => {
        it('should reject registration without email', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.1' },
                body: JSON.stringify({
                    password: 'StrongP4ss!',
                    name: 'Test User',
                }),
            });
            expect(res.status).toBe(400);
        });

        it('should reject registration with invalid email format', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.2' },
                body: JSON.stringify({
                    email: 'not-an-email',
                    password: 'StrongP4ss!',
                    name: 'Test User',
                }),
            });
            expect(res.status).toBe(400);
        });

        it('should reject registration with weak password (no uppercase)', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.3' },
                body: JSON.stringify({
                    email: 'valid@test.com',
                    password: 'weakpassword1',
                    name: 'Test User',
                }),
            });
            expect(res.status).toBe(400);
        });

        it('should reject registration with password shorter than 8 characters', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.4' },
                body: JSON.stringify({
                    email: 'valid@test.com',
                    password: 'Ab1',
                    name: 'Test User',
                }),
            });
            expect(res.status).toBe(400);
        });

        it('should reject registration without name', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.5' },
                body: JSON.stringify({
                    email: 'valid@test.com',
                    password: 'StrongP4ss!',
                }),
            });
            expect(res.status).toBe(400);
        });

        it('should reject registration with extra/unknown fields', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.6' },
                body: JSON.stringify({
                    email: 'valid@test.com',
                    password: 'StrongP4ss!',
                    name: 'Test',
                    role: 'admin', // przebicie autoryzacji!
                    isAdmin: true,
                }),
            });
            expect(res.status).toBe(400);
        });
    });

    describe('Login DTO Validation', () => {
        it('should reject login without email', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.7' },
                body: JSON.stringify({ password: 'any' }),
            });
            expect(res.status).toBe(400);
        });

        it('should reject login with invalid email format', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.8' },
                body: JSON.stringify({
                    email: 'not-email',
                    password: 'password',
                }),
            });
            expect(res.status).toBe(400);
        });

        it('should reject login without password', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.9' },
                body: JSON.stringify({ email: 'valid@test.com' }),
            });
            expect(res.status).toBe(400);
        });
    });

    describe('XSS / Injection Payloads', () => {
        it('should reject XSS payload in name field', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.10' },
                body: JSON.stringify({
                    email: 'xss@test.com',
                    password: 'StrongP4ss!',
                    name: '<script>alert("xss")</script>',
                }),
            });
            // Should either reject (400) or sanitize
            if (res.status === 201) {
                const data = await res.json();
                expect(data.name).not.toContain('<script>');
            }
        });

        it('should handle SQL injection payload in email gracefully', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.11' },
                body: JSON.stringify({
                    email: "admin' OR 1=1 --",
                    password: 'anything',
                }),
            });
            // Should reject as invalid email (400) — not expose data
            expect(res.status).toBe(400);
        });
    });

    describe('Error Response Security', () => {
        it('should not expose stack traces in error responses', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.12' },
                body: JSON.stringify({}),
            });
            const body = await res.json();
            expect(body).not.toHaveProperty('stack');
            expect(JSON.stringify(body)).not.toContain('node_modules');
        });

        it('should not reveal whether email exists on failed login', async () => {
            await new Promise(r => setTimeout(r, 350)); const res = await fetch(`${baseUrl}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.13' },
                body: JSON.stringify({
                    email: 'nonexistent-user-12345@test.com',
                    password: 'SomeP4ss!',
                }),
            });
            const body = await res.json();
            // Error message should be generic — not "User not found" vs "Wrong password"
            expect(body.message).toBe('Invalid credentials');
        });
    });
});
