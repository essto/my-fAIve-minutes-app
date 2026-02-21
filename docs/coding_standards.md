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

## 3. Strategia Testów (TDD)

### 3.1 Cykl Czerwony-Zielony-Refaktor
1. **RED:** Napisz unit test dla nowej funkcjonalności i zobacz, jak failuje.
2. **GREEN:** Napisz minimum kodu, aby test przeszedł.
3. **REFACTOR:** Uprość kod, zachowując zielone testy.

### 3.2 Zasady Testowania
- **Zasada AAA:** Arrange (Przygotuj), Act (Wykonaj), Assert (Sprawdź).
- **Izolacja:** Unit testy nie mogą dotykać bazy danych ani sieci. Używamy mocków dla Portów Wyjściowych.
- **Granularność:** 
  - `domain/**/__tests__/*.unit.test.ts`
  - `adapters/**/__tests__/*.integration.test.ts`

---

## 4. Obsługa Błędów

### 4.1 Błędy Domenowe
- Tworzymy dedykowane klasy błędów: `WeightTooLowError`, `UserAlreadyExistsError`.
- Błędy te są rzucane w Core Domain i wyłapywane przez Global Exception Filter w Adapterze REST (mapowanie na kody HTTP).

### 4.2 Logging
- Używamy ustrukturyzowanych logów JSON.
- Każdy błąd musi zawierać `context` (userId, correlationId, module).

---

## 5. Bezpieczeństwo (Security by Design)

### 5.1 Dane Medyczne
- **Zero PII in logs:** Nigdy nie logujemy nazwisk, e-maili ani surowych pomiarów w logach systemowych.
- **Encryption:** Dane wrażliwe są szyfrowane (AES-256) przed zapisem do bazy.

### 5.2 Row-Level Security (RLS)
- Nigdy nie polegamy tylko na `WHERE user_id = ...`. 
- Każde zapytanie do bazy musi być poprzedzone ustawieniem kontekstu usera: `SET LOCAL app.current_user_id = ...`.

---

## 6. Praca z AI Agentem

Jeśli jesteś AI Agentem pracującym nad tym projektem:
1. **Pamięć:** Przeczytaj `detailed_plan.md` przed rozpoczęciem pracy.
2. **Linter:** Uruchom `npm run lint` przed commitem.
3. **Review:** Jeśli kod przekracza 50 linii w jednej funkcji, podziel go.
4. **Docs:** Każdy nowy Port musi być udokumentowany w sekcji `ports/README.md` modułu.
