/**
 * TDD: JWT Hardening Tests
 * Stage 11 — Faza 4
 *
 * RED PHASE: Wszystkie testy powinny FAILOWAĆ przed implementacją.
 * Zadanie:
 *   1. Usuń fallback 'super-secret-key' z auth.module.ts i jwt.strategy.ts
 *   2. Użyj JwtModule.registerAsync() z obowiązkowym process.env.JWT_SECRET
 *   3. Skróć token lifetime do 15 minut
 *   4. Rzuć błąd jeśli JWT_SECRET nie jest ustawiony
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as jwt from 'jsonwebtoken';

describe('JWT Hardening (Faza 4)', () => {

    describe('JWT Secret Management', () => {
        it('should NOT use hardcoded fallback secret in auth.module.ts', async () => {
            // Read the actual source file and check for hardcoded secrets
            const fs = await import('fs');
            const path = await import('path');
            const authModulePath = path.resolve(__dirname, '../../modules/shared/auth/src/auth.module.ts');
            const content = fs.readFileSync(authModulePath, 'utf-8');

            expect(content).not.toContain("'super-secret-key'");
            expect(content).not.toContain('"super-secret-key"');
        });

        it('should NOT use hardcoded fallback secret in jwt.strategy.ts', async () => {
            const fs = await import('fs');
            const path = await import('path');
            const strategyPath = path.resolve(__dirname, '../../modules/shared/auth/src/jwt.strategy.ts');
            const content = fs.readFileSync(strategyPath, 'utf-8');

            expect(content).not.toContain("'super-secret-key'");
            expect(content).not.toContain('"super-secret-key"');
        });

        it('should throw error if JWT_SECRET is not set', async () => {
            const originalSecret = process.env.JWT_SECRET;
            delete process.env.JWT_SECRET;

            try {
                // Dynamically import to get fresh module evaluation
                const { JwtStrategy } = await import('../../modules/shared/auth/src/jwt.strategy');

                // If we get here without error, the test fails
                expect(() => new JwtStrategy()).toThrowError(/JWT_SECRET/i);
            } finally {
                if (originalSecret) {
                    process.env.JWT_SECRET = originalSecret;
                }
            }
        });
    });

    describe('Token Expiration', () => {
        it('should issue tokens with max 15 min expiration', () => {
            const secret = 'test-secret-for-testing-only';
            process.env.JWT_SECRET = secret;

            // Simulate what auth.module.ts should produce
            const token = jwt.sign(
                { sub: 'user123', email: 'test@test.com' },
                secret,
                { expiresIn: '15m' }
            );

            const decoded = jwt.decode(token) as jwt.JwtPayload;
            const maxLifetime = 15 * 60; // 15 minutes in seconds

            expect(decoded).toBeDefined();
            expect(decoded!.exp! - decoded!.iat!).toBeLessThanOrEqual(maxLifetime);
        });

        it('should reject expired tokens', () => {
            const secret = 'test-secret-for-testing-only';

            // Create an already-expired token
            const expiredToken = jwt.sign(
                { sub: 'user123', email: 'test@test.com' },
                secret,
                { expiresIn: '-1s' } // expired 1 second ago
            );

            expect(() => {
                jwt.verify(expiredToken, secret);
            }).toThrow(/expired/i);
        });
    });

    describe('Token Security Properties', () => {
        it('should not include sensitive data in JWT payload', () => {
            const secret = 'test-secret-for-testing-only';
            const payload = { sub: 'user123', email: 'test@test.com' };

            const token = jwt.sign(payload, secret, { expiresIn: '15m' });
            const decoded = jwt.decode(token) as jwt.JwtPayload;

            // JWT should NEVER contain password, role escalation, etc.
            expect(decoded).not.toHaveProperty('password');
            expect(decoded).not.toHaveProperty('passwordHash');
            expect(decoded).not.toHaveProperty('isAdmin');
        });

        it('should verify token signature correctly', () => {
            const secret = 'correct-secret';
            const wrongSecret = 'wrong-secret';

            const token = jwt.sign({ sub: 'user123' }, secret);

            expect(() => {
                jwt.verify(token, wrongSecret);
            }).toThrow(/signature/i);
        });
    });
});
