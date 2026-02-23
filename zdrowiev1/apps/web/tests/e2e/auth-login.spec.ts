import { test, expect } from '@playwright/test';

/**
 * US1: Autentykacja: Logowanie
 * User Story: Jako zarejestrowany użytkownik, chcę móc bezpiecznie zalogować się do aplikacji 
 * przy użyciu mojego adresu e-mail i hasła, aby uzyskać dostęp do mojego panelu zdrowia.
 */
test.describe('Auth: Login', () => {
    test('powinien pozwolić użytkownikowi zalogować się poprawnymi danymi', async ({ page }) => {
        // Wejdź na stronę logowania
        await page.goto('/login');

        // Wypełnij formularz
        await page.getByLabel('Email').fill('demo@example.com');
        await page.getByLabel('Hasło').fill('Password123!');

        // Kliknij zaloguj
        await page.click('button[type="submit"]');

        // Sprawdź czy przekierowano na dashboard
        await expect(page).toHaveURL(/.*dashboard/);

        // Sprawdź czy token jest w localStorage
        const token = await page.evaluate(() => localStorage.getItem('token'));
        expect(token).toBeTruthy();
    });

    test('powinien pokazać błąd przy błędnych danych', async ({ page }) => {
        await page.goto('/login');

        await page.getByLabel('Email').fill('wrong@example.com');
        await page.getByLabel('Hasło').fill('WrongPassword123');

        await page.click('button[type="submit"]');

        // Powinien pojawić się błąd (generalError div z CSS Modules)
        const errorMsg = page.getByText('Błąd logowania');
        await expect(errorMsg).toBeVisible({ timeout: 10000 });
    });
});
