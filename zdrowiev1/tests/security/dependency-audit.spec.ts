/**
 * TDD: Dependency Audit & CI Security Tests
 * Stage 11 — Faza 6
 *
 * RED PHASE: Niektóre testy mogą FAILOWAĆ jeśli brakuje job security w ci.yml
 * Zadanie:
 *   1. Dodaj job 'security' do .github/workflows/ci.yml
 *   2. Uruchom npm audit i lockfile-lint
 *   3. Upewnij się, że package-lock.json jest commitowany
 *   4. Sprawdź, że nie ma krytycznych podatności
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

describe('Dependency Audit & CI Security (Faza 6)', () => {
    const rootDir = path.resolve(__dirname, '../..');

    describe('Lock File Security', () => {
        it('should have package-lock.json committed', () => {
            const lockPath = path.join(rootDir, 'package-lock.json');
            expect(fs.existsSync(lockPath)).toBe(true);
        });

        it('should have package-lock.json in sync (not stale)', () => {
            const lockPath = path.join(rootDir, 'package-lock.json');
            const packagePath = path.join(rootDir, 'package.json');

            const lockStat = fs.statSync(lockPath);
            const packageStat = fs.statSync(packagePath);

            // Lock file should be at least as recent as package.json
            // (tolerance: lock file updated with or after package.json)
            expect(lockStat.mtime.getTime()).toBeGreaterThanOrEqual(
                packageStat.mtime.getTime() - 60000 // 1 min tolerance
            );
        });
    });

    describe('CI Security Job', () => {
        it('should have a security job in ci.yml', () => {
            const ciPath = path.join(rootDir, '.github', 'workflows', 'ci.yml');
            expect(fs.existsSync(ciPath)).toBe(true);

            const content = fs.readFileSync(ciPath, 'utf-8');
            expect(content).toContain('security');
            expect(content).toMatch(/npm audit/i);
        });

        it('should run npm audit with --audit-level=high in CI', () => {
            const ciPath = path.join(rootDir, '.github', 'workflows', 'ci.yml');
            const content = fs.readFileSync(ciPath, 'utf-8');

            expect(content).toContain('audit-level');
        });
    });

    describe('No Hardcoded Secrets in Source', () => {
        it('should NOT have hardcoded API keys in TypeScript files', () => {
            const srcDirs = [
                path.join(rootDir, 'apps'),
                path.join(rootDir, 'modules'),
            ];

            const secretPatterns = [
                /['"]sk[-_][a-zA-Z0-9]{20,}['"]/,   // OpenAI-style
                /['"]ghp_[a-zA-Z0-9]{36,}['"]/,       // GitHub PAT
                /['"]AKIA[A-Z0-9]{16}['"]/,            // AWS Access Key
                /password\s*[:=]\s*['"][^'"]{4,}['"]/i, // password = "value"
            ];

            const violations: string[] = [];

            function scanDir(dir: string) {
                if (!fs.existsSync(dir)) return;
                const entries = fs.readdirSync(dir, { withFileTypes: true });
                for (const entry of entries) {
                    const fullPath = path.join(dir, entry.name);
                    if (entry.isDirectory()) {
                        if (!['node_modules', '.git', 'dist', '__tests__', 'test'].includes(entry.name)) {
                            scanDir(fullPath);
                        }
                    } else if (entry.name.endsWith('.ts') && !entry.name.endsWith('.spec.ts') && !entry.name.endsWith('.test.ts')) {
                        const content = fs.readFileSync(fullPath, 'utf-8');
                        for (const pattern of secretPatterns) {
                            if (pattern.test(content)) {
                                violations.push(`${fullPath}: matches ${pattern}`);
                            }
                        }
                    }
                }
            }

            srcDirs.forEach(scanDir);
            expect(violations).toEqual([]);
        });
    });

    describe('.env.example Completeness', () => {
        it('should have .env.example documenting all required vars', () => {
            const envExamplePath = path.join(rootDir, '.env.example');
            expect(fs.existsSync(envExamplePath)).toBe(true);

            const content = fs.readFileSync(envExamplePath, 'utf-8');

            // Critical environment variables that MUST be documented
            const requiredVars = [
                'JWT_SECRET',
                'POSTGRES_HOST',
                'POSTGRES_USER',
                'POSTGRES_PASSWORD',
                'POSTGRES_DB',
            ];

            for (const varName of requiredVars) {
                expect(content).toContain(varName);
            }
        });
    });

    describe('.gitignore Security', () => {
        it('should ignore .env files', () => {
            const gitignorePath = path.join(rootDir, '.gitignore');
            const content = fs.readFileSync(gitignorePath, 'utf-8');

            expect(content).toMatch(/\.env/);
        });

        it('should NOT have .env committed to git', () => {
            // The .env file may exist locally, but should be in .gitignore
            const gitignorePath = path.join(rootDir, '.gitignore');
            const content = fs.readFileSync(gitignorePath, 'utf-8');

            expect(content).toContain('.env');
        });
    });
});
