import { test, expect } from '@playwright/test';

/**
 * US2: Autentykacja: Rejestracja
 * User Story: Jako nowa osoba, chcę móc założyć konto w aplikacji, podając mój e-mail i tworząc hasło, 
 * aby rozpocząć monitorowanie mojego zdrowia.
 */
test.describe('Auth: Register', () => {
    test('powinien pozwolić nowemu użytkownikowi zarejestrować się', async ({ page }) => {
        await page.goto('/register');

        const randomEmail = `test-${Date.now()}@example.com`;
        await page.fill('#email', randomEmail);
        await page.fill('#password', 'Password123!');

        await page.click('#register-submit');

        // Powinien przekierować na stronę logowania z parametrem registered=true
        await expect(page).toHaveURL(/.*login\?registered=true/);
    });

    test('powinien pokazać błąd gdy email jest już zajęty', async ({ page }) => {
        await page.goto('/register');

        // Używamy maila demo, który na pewno istnieje
        await page.fill('#email', 'demo@example.com');
        await page.fill('#password', 'Password123!');

        await page.click('#register-submit');

        // Powinien pojawić się błąd (selektor id="general-error")
        const errorMsg = page.locator('#general-error');
        await expect(errorMsg).toBeVisible();
    });
});
