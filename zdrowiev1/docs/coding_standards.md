# 🛠️ Standardy Kodowania: Zdrowie App

> Ten dokument jest nadrzędnym źródłem zasad dla wszystkich deweloperów (AI i ludzi). Złamanie tych zasad jest traktowane jako błąd (bug).

---

## 1. Architektura Heksagonalna (Zasady Izolacji)

### 1.1 Core Domain (Serce)
- **Zero Zależności:** Warstwa `domain/` NIE MOŻE importować niczego z platformy (np. `@nestjs/*`, `drizzle-orm`, `axios`). Używamy tylko czystego TypeScript.
- **Logika Biznesowa:** Tu mieszkają Use Case'y i Encje. Dokumentujemy tu "CO" aplikacja robi, a nie "JAK".

### 1.2 Porty (Kontrakty)
- **Definicja:** Każdy Port to interfejs TS. 
- **In-Ports:** Interfejsy dla Use Case'ów (wywoływane przez API/MCP).
- **Out-Ports:** Interfejsy dla usług zewnętrznych (DB, API, BLE). Core rozmawia z nimi, nie wiedząc, kto je implementuje.

### 1.3 Adaptery (Technologia)
- **Miejsce dla frameworków:** Tu używamy NestJS, Drizzle, bibliotek BLE.
- **Zasada:** Adapter implementuje Port. Jeśli zmienimy bazę z Postgres na MongoDB, zmieniamy tylko Adapter w `adapters/db/`.

---

## 2. Clean Code & TypeScript

### 2.1 Zasady Ogólne
- **DRY (Don't Repeat Yourself):** Wspólna logika ląduje w `@packages/shared`.
- **KISS (Keep It Simple, Stupid):** Prosty kod > Sprytny kod. Funkcja ma robić JEDNĄ rzecz.
- **YAGNI (You Ain't Gonna Need It):** Nie implementujemy funkcji "na zapas".

### 2.2 TypeScript Strict Mode
- **No Any:** Użycie `any` jest zakazane. Jeśli typ jest nieznany, używamy `unknown`.
- **Type Safety:** Każda funkcja MUSI mieć zdefiniowane typy argumentów i zwracanej wartości.
- **Zod as Guard:** Walidacja Zod na każdej granicy systemu (Request → API, DB → Core).

### 2.3 Naming (Nazewnictwo)
- **Pliki:** `kebab-case.ts` (np. `add-weight.use-case.ts`).
- **Klasy/Interfejsy:** `PascalCase` (np. `WeightRepository`).
- **Funkcje/Zmienne:** `camelCase` (np. `calculateBmi`).
- **Stałe:** `UPPER_SNAKE_CASE`.

---

## 3. Strategia Testów (Testing Standards)

> Każda nowa funkcjonalność MUSI rozpoczynać się od testu. Kod bez testu nie przejdzie przez bramkę CI/CD.

### 3.1 Hierarchia i Rodzaje Testów
| Typ Testu | Cel | Lokalizacja | Narzędzie |
|-----------|-----|-------------|-----------|
| **Unit** | Testuje czystą logikę w `domain/`. Żadnych baz, sieci, frameworków. | `domain/**/__tests__/*.unit.test.ts` | Vitest |
| **Integration** | Testuje adaptery (np. czy Repository poprawnie zapisuje w DB). | `adapters/**/__tests__/*.integration.test.ts` | Vitest + Testcontainers |
| **Contract** | Sprawdza czy Adapter poprawnie implementuje interfejs Portu. | `adapters/**/__tests__/*.contract.test.ts` | Vitest |
| **Smoke** | Błyskawiczna weryfikacja czy "system żyje" po deployu/restarcie. | `infrastructure/smoke-tests/` | Curl / Playwright |
| **E2E (Web)** | Pełny scenariusz użytkownika w przeglądarce. | `apps/web/__e2e__/*.spec.ts` | Playwright |
| **E2E (Mobile)** | Scenariusze mobilne. | `apps/mobile/__e2e__/*.spec.ts` | Detox / Expo Test |

### 3.2 Procedura Smoke Test (Post-Deployment)
Zautomatyzowany skrypt uruchamiany natychmiast po `docker compose up` lub deployu na staging/prod:
1. **Connectivity:** Czy baza i Redis odpowiadają?
2. **Health Endpoints:** Czy `/health` zwraca 200 OK?
3. **Pivotal Flow:** Czy strona logowania się ładuje? (Test trwający < 5 sekund).
4. **Auth Check:** Czy można wygenerować token testowy?

### 3.3 Procedura TDD (Test-Driven Development)
1. **Zdefiniuj Wymagania:** Zrozum co funkcja ma robić.
2. **Napisz Test Unitowy (RED):** Zakoduj oczekiwany rezultat. Test musi zawieść.
3. **Zaimplementuj Minimum Kodu (GREEN):** Napisz tylko tyle, by test przeszedł.
4. **Refaktor (REFACTOR):** Dopracuj kod, usuń duplikaty, zadbaj o czystość.
5. **Dopisz Test Integracyjny:** Jeśli funkcja dotyka bazy danych lub zewnętrznego API.

### 3.3 Definition of Done (Testing)
- Pokrycie kodu (Coverage) dla warstwy `domain` wynosi **≥90%**.
- Pokrycie kodu dla warstwy `adapters` wynosi **≥80%**.
- Wszystkie ścieżki krytyczne mają testy E2E.
- Linter i testy przechodzą lokalnie przed `git push`.

---

## 4. Obsługa Błędów

### 4.1 Błędy Domenowe
- Tworzymy dedykowane klasy błędów: `WeightTooLowError`, `UserAlreadyExistsError`.
- Błędy te są rzucane w Core Domain i wyłapywane przez Global Exception Filter w Adapterze REST (mapowanie na kody HTTP).

### 4.2 Logging
- Używamy ustrukturyzowanych logów JSON.
- Każdy błąd musi zawierać `context` (userId, correlationId, module).

---

## 5. Procedury Serwerowe (Infrastructure Standards)

> Serwery muszą startować w przewidywalny sposób. Używamy Docker Compose do zarządzania infrastrukturą.

### 5.1 Mapa Portów (Standard Port Mapping)
| Usługa | Port | Opis |
|--------|------|------|
| **PostgreSQL** | 5432 | Główna baza danych |
| **Redis** | 6379 | Cache / Sesje |
| **Web (Frontend)** | 3000 | Next.js Dev Server |
| **API (Backend)** | 3001 | NestJS API |
| **MCP Server** | 8080 | AI Maintenance Interface |

### 5.2 Kolejność Uruchamiania (lub użyj `/run-servers`)
1. **Infrastruktura:** `docker compose up -d`
2. **Backend:** `cd apps/api && npm run build && npm run start`
3. **Frontend:** `cd apps/web && npm run dev`
4. **Smoke Test:** `powershell -File scripts/check-stack.ps1`

### 5.3 Weryfikacja Startu (Health Checks)
Po uruchomieniu serwerów, AI Agent musi sprawdzić ich status:
- **Smoke test:** Uruchomienie `scripts/check-stack.ps1` weryfikuje cały stos.
- **Web:** Sprawdzenie dostępności strony głównej na porcie 3000.
- **API:** Sprawdzenie dostępności na porcie 3001.

### 5.4 Pułapki Monorepo i E2E (Krytyczne dla AI!)
> W przeszłości narobiliśmy tutaj sporo błędów. Zapamiętaj te punkty:
1. **Konflikt portów i Pętla Proxy:** Next.js ZAWSZE na 3000. NestJS ZAWSZE na 3001. Rewrite w Next.js musi wskazywać na `localhost:3001/api`. Nie twórz pętli (proxy do 3000).
2. **NestJS `start --watch` w Monorepo:** Zwykły `nest start --watch` nie zadziała przy importach poza `src/` (np. relatywne `../../modules`). Skrypt `dev` musi korzystać z pre-buildowanej wersji (`npm run build && npm run start`).
3. **Kontekst Playwright:** Testy E2E (Playwright) uruchamiaj ZAWSZE z roota monorepo (`npx playwright test`), NIE z `apps/web/`. W przeciwnym razie testy nie znajdą konfiguracji `playwright.config.ts` i `baseURL`.
4. **Playwright `baseURL`:** Musi wskazywać na frontend (port 3000), a nie na API. Zawsze o tym pamiętaj.

---

## 6. Bezpieczeństwo (Security by Design)

### 6.1 Dane Medyczne
- **Zero PII in logs:** Nigdy nie logujemy nazwisk, e-maili ani surowych pomiarów w logach systemowych.
- **Encryption:** Dane wrażliwe są szyfrowane (AES-256) przed zapisem do bazy.

### 6.2 Row-Level Security (RLS)
- Nigdy nie polegamy tylko na `WHERE user_id = ...`. 
- Każde zapytanie do bazy musi być poprzedzone ustawieniem kontekstu usera: `SET LOCAL app.current_user_id = ...`.

---

## 7. Praca z AI Agentem
 
 Jeśli jesteś AI Agentem pracującym nad tym projektem:
 1. **Pamięć:** Przeczytaj `detailed_plan.md` przed rozpoczęciem pracy.
 2. **Linter:** Uruchom `npm run lint` przed commitem.
 3. **Review:** Jeśli kod przekracza 50 linii w jednej funkcji, podziel go.
 4. **Docs:** Każdy nowy Port musi być udokumentowany w sekcji `ports/README.md` modułu.
 5. **Verify Servers:** Zawsze sprawdzaj health-checki po restarcie środowiska.
 
 ---
 
 ## 8. Zasady współpracy z GitHub (GitHub Collaboration)
 
 > **KRYTYCZNE SZEFIE:** Poniższe zasady są nienegocjowalne dla AI Agenta.
 
 ### 8.1 Autoryzacja Push
 - **Zasada "Consent-First":** Przed wykonaniem KAŻDEJ komendy `git push`, AI Agent MUSI uzyskać wyraźną zgodę użytkownika w czacie.
 - **Uzgodnienie Brancha:** AI Agent MUSI każdorazowo potwierdzić z użytkownikiem, na który branch (np. `zdrowiev1`, `master`, `feature/...`) ma trafić kod.
 - **Zero niespodzianek:** Zakaz używania flag `-f` (force) bez szczegółowego omówienia ryzyka i zgody użytkownika.
 
 ### 8.2 Atomic Commits
 - Każdy commit powinien dotyczyć jednej logicznej zmiany.
 - Wiadomości commitów muszą być zgodne z **Conventional Commits** (np. `feat:`, `fix:`, `docs:`, `chore:`).
