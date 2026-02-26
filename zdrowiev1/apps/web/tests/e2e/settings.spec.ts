import { test, expect } from '@playwright/test';

test.describe('Settings Page', () => {
    test.beforeEach(async ({ page }) => {
        // Mock authentication state so middleware doesn't redirect
        await page.context().addCookies([
            {
                name: 'auth_token',
                value: 'fake-jwt-token',
                domain: 'localhost',
                path: '/',
            }
        ]);
    });

    test('should allow user to navigate to settings and see sections', async ({ page }) => {
        await page.goto('/pl/settings');

        // Check header
        await expect(page.getByRole('heading', { name: 'Ustawienia' })).toBeVisible();

        // Check Profile section exist
        await expect(page.locator('text=Jan Kowalski')).toBeVisible();

        // Check Theme section
        const themeBtn = page.getByRole('button', { name: 'light_theme' });
        await expect(themeBtn).toBeVisible();

        // Check language select
        const langSelect = page.getByRole('combobox', { name: 'language' });
        await expect(langSelect).toHaveValue('pl');

        // Change language
        await langSelect.selectOption('en');
        await page.waitForURL('**/en/settings');

        // Assert English header is visible
        await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
    });
});
