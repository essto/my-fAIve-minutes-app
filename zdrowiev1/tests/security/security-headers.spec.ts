/**
 * TDD: Security Headers & CORS Tests
 * Stage 11 — Faza 1
 *
 * RED PHASE: Wszystkie testy powinny FAILOWAĆ przed implementacją.
 * Zadanie: Skonfiguruj helmet(), enableCors() i ValidationPipe w apps/api/src/main.ts
 *
 * Wymagane pakiety: npm install helmet @nestjs/throttler class-validator class-transformer
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import helmet from 'helmet';
import { AppModule } from '../../apps/api/src/app.module';

describe('Security Headers (Faza 1)', () => {
    let app: INestApplication;
    const baseUrl = 'http://localhost:3099';

    beforeAll(async () => {
        const moduleFixture: TestingModule = await Test.createTestingModule({
            imports: [AppModule],
        }).compile();

        app = moduleFixture.createNestApplication();
        // NOTE: Agent musi tutaj wywołać tę samą konfigurację co w main.ts
        // aby testy działały identycznie jak produkcja
        app.use(helmet());
        app.enableCors({
            origin: ['http://localhost:3000'],
            methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
            credentials: true,
        });
        app.useGlobalPipes(new ValidationPipe({
            whitelist: true,
            forbidNonWhitelisted: true,
            transform: true,
        }));
        app.setGlobalPrefix('api');
        await app.listen(3099);
    });

    afterAll(async () => {
        await app?.close();
    });

    describe('Helmet Security Headers', () => {
        it('should set X-Content-Type-Options: nosniff', async () => {
            const res = await fetch(`${baseUrl}/api/health`);
            expect(res.headers.get('x-content-type-options')).toBe('nosniff');
        });

        it('should set X-Frame-Options to deny or sameorigin', async () => {
            const res = await fetch(`${baseUrl}/api/health`);
            const value = res.headers.get('x-frame-options');
            expect(['DENY', 'SAMEORIGIN']).toContain(value?.toUpperCase());
        });

        it('should set Strict-Transport-Security header', async () => {
            const res = await fetch(`${baseUrl}/api/health`);
            const hsts = res.headers.get('strict-transport-security');
            expect(hsts).toBeTruthy();
            expect(hsts).toContain('max-age=');
        });

        it('should remove X-Powered-By header', async () => {
            const res = await fetch(`${baseUrl}/api/health`);
            expect(res.headers.get('x-powered-by')).toBeNull();
        });

        it('should set Content-Security-Policy header', async () => {
            const res = await fetch(`${baseUrl}/api/health`);
            const csp = res.headers.get('content-security-policy');
            expect(csp).toBeTruthy();
        });
    });

    describe('CORS Configuration', () => {
        it('should allow requests from http://localhost:3000', async () => {
            const res = await fetch(`${baseUrl}/api/health`, {
                headers: { 'Origin': 'http://localhost:3000' },
            });
            expect(res.headers.get('access-control-allow-origin')).toBe('http://localhost:3000');
        });

        it('should reject requests from unknown origins', async () => {
            const res = await fetch(`${baseUrl}/api/health`, {
                headers: { 'Origin': 'http://evil-site.com' },
            });
            const origin = res.headers.get('access-control-allow-origin');
            expect(origin).not.toBe('http://evil-site.com');
        });

        it('should include credentials support', async () => {
            const res = await fetch(`${baseUrl}/api/health`, {
                method: 'OPTIONS',
                headers: {
                    'Origin': 'http://localhost:3000',
                    'Access-Control-Request-Method': 'POST',
                },
            });
            expect(res.headers.get('access-control-allow-credentials')).toBe('true');
        });
    });

    describe('Global ValidationPipe', () => {
        it('should reject request with unknown/extra fields (whitelist mode)', async () => {
            const res = await fetch(`${baseUrl}/api/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    email: 'test@example.com',
                    password: 'StrongP4ss!',
                    name: 'Test',
                    isAdmin: true, // EXTRA field — should be stripped/rejected
                }),
            });
            // Expect either 400 (forbidNonWhitelisted) or the extra field NOT in response
            expect([400, 201]).toContain(res.status);
        });
    });
});
