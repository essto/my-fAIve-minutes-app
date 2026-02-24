import { test, expect } from '@playwright/test';

test.describe('Performance — Lighthouse Requirements', () => {
    test('main pages load within 3s', async ({ page }) => {
        const start = Date.now();
        await page.goto('/dashboard');
        expect(Date.now() - start).toBeLessThan(3000);
    });

    test('proper SEO meta tags', async ({ page }) => {
        await page.goto('/login');
        expect(await page.title()).toContain('Zdrowie');
        expect(await page.$('meta[name="description"]')).not.toBeNull();
        expect(await page.$('meta[name="viewport"]')).not.toBeNull();
    });

    test('images use next/image with loading attr', async ({ page }) => {
        await page.goto('/dashboard');
        const rawImgs = await page.$$('img:not([loading])');
        for (const img of rawImgs) {
            const src = await img.getAttribute('src');
            // If it's a real image and not an inline base64 or a next/image loader placeholder
            if (src && !src.startsWith('data:')) {
                expect(await img.getAttribute('loading')).toBeTruthy();
            }
        }
    });
});
