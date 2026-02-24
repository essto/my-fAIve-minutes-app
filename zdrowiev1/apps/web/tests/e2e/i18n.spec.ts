import { test, expect } from '@playwright/test';

test.describe('i18n', () => {
    test('default language is Polish', async ({ page }) => {
        await page.goto('/login');
        await expect(page.locator('html')).toHaveAttribute('lang', 'pl');
        await expect(page.getByRole('button', { name: /zaloguj/i })).toBeVisible();
    });

    test('can switch to English', async ({ page }) => {
        await page.goto('/login');

        // First, verify we're on PL
        await expect(page.locator('html')).toHaveAttribute('lang', 'pl');

        // Click language switcher
        await page.getByRole('button', { name: /Zmień język/i }).click();

        // The text on the button itself changes from "English" to "Polski" after switching
        await expect(page.locator('html')).toHaveAttribute('lang', 'en');
        await expect(page.getByRole('button', { name: /log in/i })).toBeVisible();
    });
});
