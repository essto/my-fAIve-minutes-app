/**
 * Routing Contract Tests
 * 
 * These tests verify that ALL page and component files use @/i18n/routing 
 * instead of raw next/navigation for routing utilities.
 * 
 * WHY: When useRouter/Link/redirect come from next/navigation, they don't
 * prepend the locale prefix. router.push('/dashboard') goes to /dashboard (404)
 * instead of /pl/dashboard. This is invisible in unit tests because both 
 * modules get mocked the same way.
 * 
 * This test catches it by scanning actual source files.
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

const SRC_DIR = path.resolve(__dirname, '../../src');

// These are the ONLY files allowed to import from next/navigation directly
const ALLOWED_FILES = [
    'layout.tsx',      // Server component: uses notFound(), redirect() (server-side)
    'page.tsx',        // Root page: server-side redirect only (no locale context)
    'SimpleAnalytics.tsx', // Uses useSearchParams which has no i18n equivalent
    'setupTests.ts',   // Test setup
];

// Banned imports — these MUST come from @/i18n/routing
const BANNED_PATTERNS = [
    { pattern: /import\s+.*\bfrom\s+['"]next\/navigation['"]/g, source: 'next/navigation' },
    { pattern: /import\s+.*\bfrom\s+['"]next\/link['"]/g, source: 'next/link' },
];

// Only check for routing-specific imports (useRouter, usePathname, Link, redirect)
const ROUTING_IMPORTS = /\b(useRouter|usePathname|Link|redirect)\b/;

function getAllTsxFiles(dir: string): string[] {
    const results: string[] = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            if (['node_modules', '.next', '__tests__', 'dist'].includes(entry.name)) continue;
            results.push(...getAllTsxFiles(fullPath));
        } else if (entry.name.endsWith('.tsx') || entry.name.endsWith('.ts')) {
            if (entry.name.endsWith('.test.tsx') || entry.name.endsWith('.test.ts') ||
                entry.name.endsWith('.spec.tsx') || entry.name.endsWith('.spec.ts')) continue;
            results.push(fullPath);
        }
    }
    return results;
}

describe('Routing Contract: i18n import compliance', () => {
    const files = getAllTsxFiles(SRC_DIR);

    it('should find source files to check', () => {
        expect(files.length).toBeGreaterThan(0);
    });

    it('no component/page file should import useRouter/usePathname/Link/redirect from next/navigation', () => {
        const violations: { file: string; line: number; content: string }[] = [];

        for (const file of files) {
            const basename = path.basename(file);
            if (ALLOWED_FILES.includes(basename)) continue;

            const content = fs.readFileSync(file, 'utf-8');
            const lines = content.split('\n');

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                for (const banned of BANNED_PATTERNS) {
                    if (banned.pattern.test(line) && ROUTING_IMPORTS.test(line)) {
                        violations.push({
                            file: path.relative(SRC_DIR, file),
                            line: i + 1,
                            content: line.trim(),
                        });
                    }
                    banned.pattern.lastIndex = 0; // reset regex state
                }
            }
        }

        if (violations.length > 0) {
            const msg = violations
                .map(v => `  ❌ ${v.file}:${v.line}\n     ${v.content}`)
                .join('\n');
            expect.fail(
                `Found ${violations.length} file(s) importing routing utilities from wrong module.\n` +
                `Use '@/i18n/routing' instead of 'next/navigation' or 'next/link'.\n\n${msg}`
            );
        }
    });

    it('no component/page file should import Link from next/link', () => {
        const violations: { file: string; line: number; content: string }[] = [];

        for (const file of files) {
            const basename = path.basename(file);
            if (ALLOWED_FILES.includes(basename)) continue;

            const content = fs.readFileSync(file, 'utf-8');
            const lines = content.split('\n');

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                if (/import\s+.*\bfrom\s+['"]next\/link['"]/.test(line)) {
                    violations.push({
                        file: path.relative(SRC_DIR, file),
                        line: i + 1,
                        content: line.trim(),
                    });
                }
            }
        }

        if (violations.length > 0) {
            const msg = violations
                .map(v => `  ❌ ${v.file}:${v.line}\n     ${v.content}`)
                .join('\n');
            expect.fail(
                `Found ${violations.length} file(s) importing Link from 'next/link'.\n` +
                `Use { Link } from '@/i18n/routing' instead.\n\n${msg}`
            );
        }
    });
});
