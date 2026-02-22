import { test, expect } from '@playwright/test';

/**
 * US-NOTIF: Panel Powiadomień
 * Jako zalogowany użytkownik, chcę widzieć swoje powiadomienia w dashboardzie,
 * aby być na bieżąco z alertami zdrowotnymi.
 */
test.describe('Notifications: Panel', () => {
    test.beforeEach(async ({ page }) => {
        // Zaloguj się przed testem (używając demo usera)
        await page.goto('/login');
        await page.getByLabel('Email').fill('demo@example.com');
        await page.getByLabel('Hasło').fill('Password123!');
        await page.getByRole('button', { name: 'Zaloguj' }).click();
        // Czekaj na dashboard (dłuższy timeout na cold-start Next.js compilation)
        await expect(page).toHaveURL(/.*dashboard/, { timeout: 15000 });
    });

    test('powinien wyświetlić dzwonek powiadomień na dashboardzie', async ({ page }) => {
        const bell = page.getByTestId('notification-bell');
        await expect(bell).toBeVisible();
    });

    test('powinien otworzyć panel powiadomień po kliknięciu dzwonka', async ({ page }) => {
        await page.click('#notification-bell-button');

        const panel = page.getByTestId('notification-panel');
        await expect(panel).toBeVisible();
        await expect(panel).toContainText('Powiadomienia');
    });

    test('powinien pokazać panel z powiadomieniami lub stan pusty', async ({ page }) => {
        await page.click('#notification-bell-button');
        const panel = page.getByTestId('notification-panel');
        await expect(panel).toBeVisible();
    });

    test('powinien zamknąć panel po ponownym kliknięciu dzwonka', async ({ page }) => {
        // Otwórz panel
        await page.click('#notification-bell-button');
        await expect(page.getByTestId('notification-panel')).toBeVisible();

        // Zamknij panel
        await page.click('#notification-bell-button');
        await expect(page.getByTestId('notification-panel')).not.toBeVisible();
    });
});
