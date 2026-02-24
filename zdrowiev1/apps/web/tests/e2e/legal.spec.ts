import { test, expect } from '@playwright/test';

test.describe('Legal Pages', () => {
    test('privacy policy renders with GDPR sections', async ({ page }) => {
        await page.goto('http://localhost:3000/privacy-policy');
        await expect(page.getByRole('heading', { name: /polityka prywatności|privacy/i })).toBeVisible();
        await expect(page.getByText(/dane osobowe|RODO/i)).toBeVisible();
    });

    test('terms of service renders', async ({ page }) => {
        await page.goto('http://localhost:3000/terms');
        await expect(page.getByRole('heading', { name: /regulamin|terms/i })).toBeVisible();
    });

    test('footer has legal links', async ({ page }) => {
        await page.goto('http://localhost:3000/privacy-policy');
        const privacyLink = page.getByRole('link', { name: 'Polityka Prywatności' });
        const termsLink = page.getByRole('link', { name: 'Regulamin' });
        await expect(privacyLink).toBeVisible();
        await expect(termsLink).toBeVisible();
    });
});
