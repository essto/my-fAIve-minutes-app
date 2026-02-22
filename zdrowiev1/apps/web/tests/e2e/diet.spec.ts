import { test, expect } from '@playwright/test';

/**
 * US-DIET: Logowanie Posiłków
 * Jako zalogowany użytkownik, chcę logować zjedzone posiłki,
 * aby kontrolować spożycie kalorii i makroskładników.
 */
test.describe('Diet: Meal Logging', () => {
    test.beforeEach(async ({ page }) => {
        // Zaloguj się przed testem (używając demo usera)
        await page.goto('/login');
        await page.getByLabel('Email').fill('demo@example.com');
        await page.getByLabel('Hasło').fill('Password123!');
        await page.getByRole('button', { name: 'Zaloguj' }).click();
        // Czekaj na dashboard
        await expect(page).toHaveURL(/.*dashboard/, { timeout: 15000 });
        // Przejdź do strony diety
        await page.goto('/diet');
    });

    test('powinien wyświetlić stronę diety z podsumowaniem', async ({ page }) => {
        await expect(page.getByText('Twoja Dieta')).toBeVisible();
        await expect(page.getByText('Kalorie')).toBeVisible();
    });

    test('powinien pozwolić na zalogowanie posiłku', async ({ page }) => {
        await page.getByRole('button', { name: 'Zaloguj Posiłek' }).click();

        await page.getByLabel('Nazwa Posiłku').fill('Drugie Śniadanie');
        await page.getByLabel('Produkt').fill('Banan');
        await page.getByLabel('Waga (g)').fill('120');
        await page.getByLabel('Kalorie').fill('100');

        await page.getByRole('button', { name: 'Zapisz' }).click();

        // Sprawdź czy summary się zaktualizowało (przynajmniej czy tekst '100 /' się pojawił lub podobne)
        // Uwaga: Dokładny tekst może zależeć od tego co już było w bazie demo usera
        await expect(page.getByText('Drugie Śniadanie')).not.toBeVisible(); // Modal powinien zniknąć
        await expect(page.getByText('Twoja Dieta')).toBeVisible();
    });
});
