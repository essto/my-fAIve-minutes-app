# 📋 Szczegółowy Plan Implementacji: Zdrowie App

> **Cel:** Plan krok-po-kroku dla AI agenta, który będzie budował aplikację od zera. Każdy etap jest samodzielny i weryfikowalny. Moduły ≤1500 LOC.

---

## Spis Treści
1. [Mapa Modułów](#1-mapa-modułów)
2. [Etap 0: Scaffolding](#etap-0-scaffolding--konfiguracja)
3. [Etap 1: Shared Core](#etap-1-shared-core)
4. [Etap 2: Moduł Health](#etap-2-moduł-health)
5. [Etap 3: Moduł Diet](#etap-3-moduł-diet)
6. [Etap 4: Moduł Diagnosis](#etap-4-moduł-diagnosis)
7. [Etap 5: Moduł OCR](#etap-5-moduł-ocr)
8. [Etap 6: Moduł Visualization](#etap-6-moduł-visualization)
9. [Etap 7: Frontend Web](#etap-7-frontend-web-nextjs)
10. [Etap 8: Frontend Mobile](#etap-8-frontend-mobile-react-native)
11. [Etap 9: MCP Servers](#etap-9-mcp-servers)
12. [Etap 10: CI/CD & Deploy](#etap-10-cicd--deploy)
13. [Etap 11: Security Hardening](#etap-11-security-hardening)
14. [Etap 12: Polish & Launch](#etap-12-polish--launch)

---

## 1. Mapa Modułów

Każdy moduł to **niezależna jednostka ≤1500 LOC** z własnym core, portami i adapterami.

```
Zdrowie_v1/
├── apps/
│   ├── api/                  ← NestJS Backend (orchestrator, ~800 LOC)
│   ├── web/                  ← Next.js 15 Frontend (~1200 LOC per page)
│   └── mobile/               ← React Native / Expo (~1200 LOC per screen)
│
├── modules/
│   ├── shared/               ← Auth, User, Consent, Config
│   │   ├── auth/             ≤1500 LOC — JWT, OAuth, Guards
│   │   ├── user/             ≤1200 LOC — Profile, CRUD
│   │   ├── consent/          ≤800 LOC  — Consent CRUD + Guard
│   │   └── database/         ≤600 LOC  — Drizzle config, migrations, RLS
│   │
│   ├── health/               ← Waga, Tętno, Sen, Aktywność
│   │   ├── weight/           ≤1200 LOC — Pomiary, trend, BMI
│   │   ├── heart-rate/       ≤800 LOC  — BPM, typy, anomalie
│   │   ├── sleep/            ≤1000 LOC — Fazy, scoring, trend
│   │   └── activity/         ≤800 LOC  — Kroki, kalorie, treningi
│   │
│   ├── diet/                 ← Posiłki, Kalorie, Makro
│   │   ├── meal-log/         ≤1200 LOC — CRUD posiłków, barcode
│   │   ├── nutrition-calc/   ≤800 LOC  — Makro, mikro, deficyty
│   │   └── food-db/          ≤600 LOC  — Adapter do bazy żywności
│   │
│   ├── diagnosis/            ← Objawy, Triage, Raporty
│   │   ├── symptom-checker/  ≤1200 LOC — Zbieranie, matching
│   │   ├── triage/           ≤800 LOC  — Ocena ryzyka, rekomendacje
│   │   └── report-gen/       ≤600 LOC  — PDF dla lekarza
│   │
│   ├── ocr/                  ← Skanowanie dokumentów medycznych
│   │   ├── scanner/          ≤1000 LOC — Upload, preprocessing
│   │   ├── parser/           ≤1200 LOC — OCR + AI extraction
│   │   └── validator/        ≤600 LOC  — Weryfikacja wyników
│   │
│   ├── visualization/        ← Wykresy, Dashboardy, Raporty
│   │   ├── charts/           ≤1200 LOC — Konfiguracje wykresów
│   │   ├── dashboard/        ≤1000 LOC — Kompozycja widoków
│   │   └── export/           ≤800 LOC  — PDF/PNG generowanie
│   │
│   └── integrations/         ← Adaptery urządzeń
│       ├── withings/         ≤600 LOC
│       ├── garmin/           ≤600 LOC
│       ├── apple-health/     ≤600 LOC
│       ├── google-fit/       ≤600 LOC
│       ├── ble-scale/        ≤800 LOC
│       └── csv-import/       ≤400 LOC
│
├── infrastructure/
│   ├── mcp-servers/          ← MCP Servers dla AI maintenance
│   │   ├── health-mcp/      ≤800 LOC
│   │   ├── diet-mcp/        ≤600 LOC
│   │   ├── diagnosis-mcp/   ≤600 LOC
│   │   └── logger-mcp/      ≤400 LOC
│   └── docker/               ← Docker + compose
│
├── packages/
│   ├── design-tokens/        ≤300 LOC — kolory, fonty, spacing
│   ├── api-client/           ≤800 LOC — shared REST/WS client
│   ├── zod-schemas/          ≤600 LOC — wszystkie schematy walidacji
│   └── shared-types/         ≤400 LOC — typy TS
│
└── config/
    ├── vitest.config.ts
    ├── playwright.config.ts
    └── drizzle.config.ts
```

---

## Etap 0: Scaffolding & Konfiguracja

> **Cel:** Przygotować repozytorium, narzędzia, monorepo. Zero logiki biznesowej.
> **Szac. czas:** 1 sesja AI

### Kroki:

#### 0.1 Inicjalizacja monorepo
```bash
# Turborepo (monorepo manager)
npx -y create-turbo@latest ./ --skip-install
npm install
```
- [ ] **Plik:** `turbo.json` — pipeline: build, test, lint, dev
- [ ] **Plik:** `package.json` — workspaces: `apps/*`, `modules/*`, `packages/*`, `infrastructure/*`

#### 0.2 Konfiguracja TypeScript
- [ ] **Plik:** `tsconfig.base.json` — strict mode, paths aliases
- [ ] Każdy moduł dziedziczy z `tsconfig.base.json`
- [ ] Path aliasy: `@modules/health/*`, `@packages/schemas/*`, itd.

#### 0.3 Konfiguracja narzędzi jakości
- [ ] **ESLint:** `eslint.config.mjs` z regułami: no-console, no-any, max-lines-per-function (50)
- [ ] **Prettier:** `.prettierrc` — spójne formatowanie
- [ ] **Husky + lint-staged:** pre-commit hook → lint + unit tests
- [ ] **Commitlint:** conventional commits (feat:, fix:, test:, docs:)

#### 0.4 Konfiguracja testów
- [ ] **Vitest:** `vitest.config.ts` — coverage thresholds: core ≥90%, adapters ≥80%
- [ ] **Playwright:** `playwright.config.ts` — E2E web tests
- [ ] **Testcontainers:** setup dla PostgreSQL w testach integracyjnych

#### 0.5 Docker
- [ ] **Plik:** `docker-compose.yml` — PostgreSQL 16 + Redis 7
- [ ] **Plik:** `docker-compose.dev.yml` — hot reload, volumes
- [ ] `.env.example` — wszystkie zmienne środowiskowe

#### 0.6 Verify
```bash
npm run build   # ← powinno przejść (0 modułów)
npm run test    # ← powinno przejść (0 testów)
npm run lint    # ← powinno przejść
docker compose up -d  # ← PG + Redis startują
```

> **✅ Definition of Done:** Monorepo buduje się, linter działa, Docker podnosi PG+Redis.

---

## Etap 1: Shared Core

> **Cel:** Auth, User, Consent, Database — fundament aplikacji.
> **Szac. czas:** 2-3 sesje AI
> **Zależności:** Etap 0

### 1.1 Packages (shared)
- [ ] `packages/zod-schemas/` — wszystkie schematy z architecture_analysis.md §8
  - UserSchema, ConsentSchema, WeightReadingSchema, HeartRateSchema, SleepRecordSchema, MealEntrySchema, SymptomReportSchema
  - **Test:** `schemas.unit.test.ts` — walidacja poprawnych/niepoprawnych danych
- [ ] `packages/shared-types/` — typy TS generowane z Zod: `z.infer<typeof Schema>`
- [ ] `packages/design-tokens/` — JSON z kolorami, fontami, spacing
- [ ] `packages/api-client/` — base Axios/fetch wrapper z interceptorami (auth, retry, error)

### 1.2 Database Module (`modules/shared/database/`)
- [ ] Drizzle ORM config + connection pool
- [ ] Migration: `001_create_users.sql`
- [ ] Migration: `002_create_consents.sql`
- [ ] Migration: `003_create_audit_logs.sql`
- [ ] Migration: `004_create_device_connections.sql`
- [ ] Migration: `005_enable_rls_policies.sql` — Row-Level Security
- [ ] **Test (integration):** `database.integration.test.ts` z Testcontainers

### 1.3 Auth Module (`modules/shared/auth/`)

```
modules/shared/auth/
├── domain/
│   ├── entities/
│   │   └── auth-token.entity.ts        # Token value object
│   └── use-cases/
│       ├── login.use-case.ts           # Email/password → JWT
│       ├── register.use-case.ts        # Registration + hash
│       ├── refresh-token.use-case.ts   # Refresh flow
│       └── oauth-login.use-case.ts     # Google/Apple OAuth
├── ports/
│   ├── in/
│   │   └── auth.port.ts               # Input interface
│   └── out/
│       ├── user-repository.port.ts     # DB access
│       └── token-service.port.ts       # JWT signing/verification
├── adapters/
│   ├── rest/
│   │   └── auth.controller.ts          # REST endpoints
│   ├── jwt/
│   │   └── jose-token.adapter.ts       # jose library implementation
│   └── __tests__/
│       ├── auth.integration.test.ts
│       └── auth-port.contract.test.ts
└── domain/__tests__/
    ├── login.unit.test.ts              # TDD: pisany PRZED login.use-case.ts
    ├── register.unit.test.ts
    └── refresh-token.unit.test.ts
```

- [ ] **TDD:** Napisz `login.unit.test.ts` → potem `login.use-case.ts`
- [ ] **TDD:** Napisz `register.unit.test.ts` → potem `register.use-case.ts`
- [ ] JWT Access Token (15min) + Refresh Token (30 dni, HttpOnly cookie)
- [ ] Password hashing: bcrypt (cost=12)
- [ ] Guards: `JwtGuard`, `RolesGuard`, `TenantGuard`
- [ ] Rate limiting na login: 5 prób/min

### 1.4 User Module (`modules/shared/user/`)
- [ ] CRUD profilu (wiek, wzrost, płeć, cele)
- [ ] Soft-delete (GDPR right to erasure)
- [ ] **TDD:** Testy najpierw dla create/update/delete

### 1.5 Consent Module (`modules/shared/consent/`)
- [ ] Granularne zgody per kategoria danych (weight, heart_rate, sleep, diet, diagnosis)
- [ ] ConsentGuard — middleware sprawdzający zgody przed dostępem
- [ ] grant / revoke / list consents
- [ ] Audit log przy każdym zdarzeniu

### 1.6 API App (`apps/api/`)
- [ ] NestJS bootstrap z modułami shared
- [ ] Global exception filter (structured JSON errors)
- [ ] Swagger/OpenAPI generation
- [ ] Health check endpoint: `GET /health`

### 1.7 Verify
```bash
npm run test -- --filter=shared       # Unit + Integration
curl http://localhost:3000/health      # → { "status": "ok" }
curl -X POST /auth/register            # → 201 + JWT
curl -X POST /auth/login               # → 200 + JWT
curl -H "Authorization: Bearer ..." /users/me  # → 200 + profile
```

> **✅ DoD:** Rejestracja, login, profil, zgody działają. RLS aktywne. Coverage core ≥90%.

---

## Etap 2: Moduł Health

> **Cel:** Waga, tętno, sen, aktywność — zapis + odczyt + prosty trend.
> **Zależności:** Etap 1

### 2.1 Weight Sub-module (`modules/health/weight/`)

```
weight/
├── domain/
│   ├── entities/weight-reading.entity.ts
│   ├── use-cases/
│   │   ├── add-weight.use-case.ts
│   │   ├── get-weight-history.use-case.ts
│   │   ├── calculate-bmi.use-case.ts
│   │   └── analyze-weight-trend.use-case.ts
│   └── __tests__/                          # ← TDD: NAJPIERW testy
│       ├── add-weight.unit.test.ts
│       ├── calculate-bmi.unit.test.ts
│       └── analyze-weight-trend.unit.test.ts
├── ports/
│   ├── in/weight.port.ts
│   └── out/weight-repository.port.ts
├── adapters/
│   ├── rest/weight.controller.ts
│   ├── db/weight-drizzle.repository.ts
│   └── __tests__/
│       ├── weight-repo.integration.test.ts
│       └── weight-port.contract.test.ts
```

- [ ] **TDD:** `calculate-bmi.unit.test.ts` → `calculate-bmi.use-case.ts`
- [ ] **TDD:** `analyze-weight-trend.unit.test.ts` → `analyze-weight-trend.use-case.ts`
- [ ] Trend: linear regression na ostatnich 30 dniach
- [ ] Migration: `006_create_weight_readings.sql` (TimescaleDB hypertable)
- [ ] REST: `POST /health/weight`, `GET /health/weight?days=30`
- [ ] Zod validation na wejściu (min 0.5kg, max 500kg)

### 2.2 Heart Rate Sub-module (`modules/health/heart-rate/`)
- [ ] CRUD pomiarów tętna (resting, active, sleep, recovery)
- [ ] Wykrywanie anomalii (BPM > 120 w spoczynku = alert)
- [ ] Migration: `007_create_heart_rate_readings.sql`

### 2.3 Sleep Sub-module (`modules/health/sleep/`)
- [ ] Zapis faz snu (deep, light, REM, awake)
- [ ] Sleep score calculation (0-100)
- [ ] Migration: `008_create_sleep_records.sql`

### 2.4 Activity Sub-module (`modules/health/activity/`)
- [ ] Kroki, kalorie, dystans, treningi
- [ ] Migration: `009_create_activity_records.sql`

### 2.5 Verify
```bash
npm run test -- --filter=health
# POST weight → GET history → verify trend calculation
# POST heart-rate → verify anomaly detection
# RLS test: user A nie widzi danych user B
```

> **✅ DoD:** Zapis i odczyt zdrowie. Trend wagi działa. RLS chroni dane. Coverage ≥90%.

---

## Etap 3: Moduł Diet

> **Cel:** Logowanie posiłków, kalorie, makro, baza żywności.
> **Zależności:** Etap 1

### 3.1 Meal Log (`modules/diet/meal-log/`)
- [ ] CRUD posiłków (śniadanie/obiad/kolacja/snack)
- [ ] Dodawanie produktów z listy lub skanera kodów
- [ ] Dzienne podsumowanie kalorii i makro
- [ ] Migration: `010_create_meal_entries.sql`

### 3.2 Nutrition Calc (`modules/diet/nutrition-calc/`)
- [ ] Kalkulacja sum kalorii/białko/tłuszcz/węgle
- [ ] Wykrywanie deficytów (np. za mało białka)
- [ ] Cele dzienne per użytkownik

### 3.3 Food DB Adapter (`modules/diet/food-db/`)
- [ ] Adapter do Open Food Facts API (darmowe)
- [ ] Lokalne cache produktów w Redis
- [ ] Obsługa barcode lookup

### 3.4 Verify
```bash
# POST meal → verify daily summary
# Barcode scan → product found
# Deficit alert when protein < target
```

> **✅ DoD:** Logowanie posiłków działa. Barcode lookup łączy się z bazą. Deficyty wykrywane.

---

## Etap 4: Moduł Diagnosis

> **Cel:** Zbieranie objawów, triage AI, generowanie raportów PDF.
> **Zależności:** Etap 1

### 4.1 Symptom Checker (`modules/diagnosis/symptom-checker/`)
- [ ] Formularz objawów (nazwa, obszar ciała, intensywność, czas trwania)
- [ ] Matching chorób na podstawie objawów (algorytm + opcjonalnie AI)
- [ ] Migration: `011_create_symptom_reports.sql`, `012_create_conditions.sql`

### 4.2 Triage (`modules/diagnosis/triage/`)
- [ ] Ocena ryzyka: 🟢 zostań w domu / 🟡 lekarz / 🔴 SOR
- [ ] Uwzględnianie historii zdrowia i leków

### 4.3 Report Gen (`modules/diagnosis/report-gen/`)
- [ ] Generowanie PDF z historią objawów dla lekarza
- [ ] Użycie PDFKit lub Puppeteer

### 4.4 Verify
```bash
# Submit symptoms → get triage result
# Generate PDF → verify content
```

> **✅ DoD:** Symptom checker + triage + PDF. Dane objawów izolowane per user.

---

## Etap 5: Moduł OCR

> **Cel:** Skanowanie i interpretowanie dokumentów medycznych (wyniki badań, recepty).
> **Zależności:** Etap 1

### 5.1 Scanner (`modules/ocr/scanner/`)
- [ ] Upload plików: JPG, PNG, PDF (max 10MB)
- [ ] Preprocessing: deskew, denoise, contrast enhancement (Sharp)
- [ ] Konwersja PDF → obrazy (pdf-to-img lub Poppler)
- [ ] Migration: `013_create_scanned_documents.sql`

### 5.2 Parser (`modules/ocr/parser/`)
- [ ] **OCR Engine:** Tesseract.js (local) + opcjonalnie Google Vision API (cloud)
- [ ] **AI Extraction:** LLM (via OpenRouter) do interpretacji tekstu medycznego
  - Input: surowy tekst OCR
  - Output: ustrukturyzowane dane (Zod schema):
    ```typescript
    const MedicalDocumentSchema = z.object({
      documentType: z.enum(['blood_test', 'prescription', 'referral', 'imaging', 'other']),
      patientName: z.string().optional(),
      testDate: z.date().optional(),
      results: z.array(z.object({
        parameter: z.string(),      // np. "Hemoglobina"
        value: z.number(),          // np. 14.5
        unit: z.string(),           // np. "g/dL"
        referenceRange: z.string(), // np. "12.0-16.0"
        status: z.enum(['normal', 'low', 'high', 'critical']),
      })),
      medications: z.array(z.object({
        name: z.string(),
        dosage: z.string(),
        frequency: z.string(),
      })).optional(),
      rawText: z.string(),
    });
    ```
- [ ] Confidentiality: tekst przetwarzany lokalnie, do LLM wysyłamy bez danych osobowych (anonimizacja)

### 5.3 Validator (`modules/ocr/validator/`)
- [ ] Użytkownik weryfikuje wyextraktowane dane przed zapisem
- [ ] Flagowanie niskiej pewności OCR (confidence < 80%)
- [ ] Manualny override dla błędnie rozpoznanych wartości

### 5.4 Verify
```bash
# Upload blood test PDF → OCR → parsed results
# Verify: hemoglobin value correctly extracted
# Verify: low confidence items flagged
# Verify: personal data anonymized before LLM call
```

> **✅ DoD:** Upload → OCR → strukturyzowane dane. Anonimizacja działa. User może poprawiać.

---

## Etap 6: Moduł Visualization

> **Cel:** Piękne, interaktywne wykresy zdrowotne + eksport.
> **Zależności:** Etap 2, 3, 4

### 6.1 Charts Config (`modules/visualization/charts/`)
Biblioteka: **Recharts** (web) + **Victory Native** (mobile)

Typy wykresów:

| Typ wykresu | Dane | Zastosowanie |
|-------------|------|-------------|
| **Line Chart** (trend) | Waga w czasie | Trend spadku/wzrostu |
| **Area Chart** (stacked) | Fazy snu | Deep/REM/Light/Awake |
| **Bar Chart** (grouped) | Makro dzienne | Białko/Tłuszcz/Węgle |
| **Radar Chart** | Health Score | Ogólna ocena zdrowia |
| **Gauge** | BMI / Body Fat | Aktualna wartość vs norma |
| **Heatmap** | Aktywność | Kalendarz kroków (GitHub-style) |
| **Scatter Plot** | Korelacja | Waga vs kalorie |
| **Progress Ring** | Cele | Zamykanie pierścieni (Activity Rings) |
| **Sparkline** | Mini trend | Szybki podgląd w karcie |
| **Candlestick** | Waga dzień | Min/max/avg wagi w tygodniu |

- [ ] Chart config factory: `createChart(type, data, options) → React component`
- [ ] Responsive: mobile (uproszczony) vs desktop (pełny)
- [ ] Animacje: smooth transitions przy zmianie zakresu dat
- [ ] Ciemny/jasny motyw z design tokens

### 6.2 Dashboard Compositor (`modules/visualization/dashboard/`)
- [ ] Dashboard builder: układ kafelków drag & drop (opcja)
- [ ] Preset layouty: "Przegląd dzienny", "Analiza tygodniowa", "Raport miesięczny"
- [ ] Health Score: agregat 0-100 z wagi, snu, aktywności, diety
- [ ] Alerts: karty z anomaliami (wysoki BPM, brak snu, deficyt kalorii)

### 6.3 Export (`modules/visualization/export/`)
- [ ] Export dashboard → PDF (z wykresami)
- [ ] Export danych → CSV
- [ ] Export wykresu → PNG/SVG
- [ ] Share link (publiczny, tymczasowy) dla lekarza

### 6.4 Verify
```bash
# Render dashboard with sample data → screenshot test
# Export PDF → verify charts are embedded
# Mobile vs Desktop → verify responsive layout
# Dark/Light mode → verify colors from design tokens
```

> **✅ DoD:** ≥8 typów wykresów. Dashboard renderuje. Export PDF/PNG/CSV działa. Responsywny.

---

## Etap 7: Frontend Web (Next.js)

> **Cel:** Aplikacja webowa z pełnym UI.
> **Zależności:** Etap 1-6

### 7.1 Setup
- [ ] Next.js 15 (App Router) w `apps/web/`
- [ ] Design system z design tokens
- [ ] Layout: Sidebar + Main content area
- [ ] Auth pages: Login, Register, Forgot Password

### 7.2 Strony

| Strona | Opis | Kluczowe elementy |
|--------|------|-------------------|
| `/dashboard` | Główny pulpit | Health Score, Activity Rings, sparkliny, alerts |
| `/health/weight` | Zarządzanie wagą | Line chart trend, BMI gauge, tabela pomiarów |
| `/health/heart` | Tętno | Area chart, spoczynek vs aktywność |
| `/health/sleep` | Sen | Stacked area (fazy), sleep score |
| `/health/activity` | Aktywność | Heatmap, kroki bar chart |
| `/diet` | Dieta | Dzienne makro bars, lista posiłków, barcode scan |
| `/diagnosis` | Diagnoza | Formularz objawów, body map, historia |
| `/ocr` | Skanowanie | Upload zone, preview OCR, edycja wyników |
| `/reports` | Raporty | Generator raportów, export PDF |
| `/settings` | Ustawienia | Profil, urządzenia, zgody, theme |

### 7.3 Kluczowe wymagania UI
- [ ] **Dark/Light mode** — przełącznik + system preference
- [ ] **Responsywny** — desktop (sidebar) → tablet (collapsed) → mobile (bottom nav)
- [ ] **Animacje** — Framer Motion: page transitions, chart animations
- [ ] **Skeleton loading** — ładne loading states
- [ ] **Error boundaries** — graceful error handling per sekcja
- [ ] **a11y** — ARIA labels, keyboard navigation, contrast ≥4.5:1

### 7.4 Verify (Playwright E2E)
```bash
# Login → Dashboard renders → Click Weight → Chart loads
# Upload OCR document → verify results displayed
# Export PDF → verify file downloaded
# Mobile viewport → verify bottom nav
```

> **✅ DoD:** Wszystkie strony renderują. E2E ≥10 krytycznych ścieżek. WCAG AA.

---

## Etap 8: Frontend Mobile (React Native)

> **Cel:** Natywna aplikacja iOS/Android z BLE.
> **Zależności:** Etap 1-6

### 8.1 Setup
- [ ] React Native + Expo w `apps/mobile/`
- [ ] Shared hooks i api-client z `packages/`
- [ ] Navigation: React Navigation (bottom tabs + stack)

### 8.2 Ekrany
- [ ] **Home** — mini dashboard, Activity Rings, dzienny podgląd
- [ ] **Health** — weight, heart, sleep cards z sparklinami
- [ ] **Diet** — szybkie logowanie posiłku, barcode scanner (kamera)
- [ ] **OCR** — skanowanie aparatem + gallery upload
- [ ] **Profile** — ustawienia, urządzenia, zgody

### 8.3 BLE Integration
- [ ] `react-native-ble-plx` — skanowanie wag BLE
- [ ] Pairing flow z wagi Bluetooth
- [ ] Auto-sync weight reading po pomiarze

### 8.4 Verify
```bash
# Detox E2E: Login → Home renders → Log meal → Verify
# BLE mock test: simulated scale → weight recorded
```

> **✅ DoD:** Ekrany renderują. BLE pairing flow działa. Barcode scanner działa.

---

## Etap 9: MCP Servers

> **Cel:** Serwery MCP dla AI maintenance.
> **Zależności:** Etap 1-6

### 9.1 Health MCP (`infrastructure/mcp-servers/health-mcp/`)
- [ ] Tools: `get_weight_history`, `add_weight_reading`, `get_health_score`
- [ ] Resources: `weight_trend`, `sleep_summary`

### 9.2 Diet MCP (`infrastructure/mcp-servers/diet-mcp/`)
- [ ] Tools: `log_meal`, `get_daily_summary`, `search_food`

### 9.3 Diagnosis MCP (`infrastructure/mcp-servers/diagnosis-mcp/`)
- [ ] Tools: `submit_symptoms`, `get_triage`, `generate_report`

### 9.4 Logger MCP (`infrastructure/mcp-servers/logger-mcp/`)
- [ ] Tools: `get_recent_errors`, `get_error_patterns`, `suggest_fix`
- [ ] Auto-categorization: severity, module, frequency

### 9.5 Verify
```bash
# Connect MCP client → list tools → call get_weight_history
# Logger MCP: inject error → verify it appears in get_recent_errors
```

> **✅ DoD:** 4 MCP servers. Każdy responds do tool calls. Logger zbiera i kategoryzuje.

---

## Etap 10: CI/CD & Deploy

> **Cel:** Automatyzacja build/test/deploy.
> **Zależności:** Etap 0-9

### 10.1 GitHub Actions

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  lint:     # ESLint + Prettier check
  unit:     # Vitest unit tests (coverage gate)
  integration:  # Vitest + Testcontainers
  e2e-web:      # Playwright
  e2e-mobile:   # Detox (opcja: na self-hosted runner)
  security:     # npm audit + custom RLS tests
  build:        # Turborepo build
  deploy:       # Only on main
```

- [ ] Coverage gates: core ≥90%, adapters ≥80%
- [ ] Auto-deploy to staging on PR merge
- [ ] Production deploy on tag/release

### 10.2 Docker Production
- [ ] Multi-stage Dockerfile (build → runtime)
- [ ] Image size < 200MB
- [ ] Health checks

### 10.3 Verify
```bash
# Push to branch → CI runs all jobs → green
# Merge to main → auto-deploy to staging
```

> **✅ DoD:** CI/CD pipeline green. Auto-deploy działa. Coverage gates blokują niską jakość.

---

## Etap 11: Security Hardening

> **Cel:** Finalne testy bezpieczeństwa.
> **Zależności:** Etap 1-10

- [ ] **OWASP Top 10** scan (ZAP lub equivalent)
- [ ] **RLS stress test:** 1000 zapytań cross-user → 0 leaks
- [ ] **JWT security:** token rotation, blacklist on logout
- [ ] **Rate limiting:** 100 req/min per user, 5 login attempts/min
- [ ] **CORS:** whitelisted domains only
- [ ] **Helmet.js:** security headers (CSP, HSTS, X-Frame-Options)
- [ ] **SQL injection test:** SQLMap scan
- [ ] **Dependency audit:** `npm audit --production` → 0 critical

> **✅ DoD:** 0 critical/high vulnerabilities. RLS leak test passed. Penetration test clean.

---

## Etap 12: Polish & Launch

> **Cel:** Ostateczne poprawki, dokumentacja, launch.
> **Zależności:** Etap 0-11

- [ ] **Performance:** Lighthouse score ≥90 (Performance, A11y, BP, SEO)
- [ ] **i18n:** PL + EN (react-intl)
- [ ] **Onboarding:** First-time user wizard (profil, cele, urządzenia)
- [ ] **Documentation:** README, API docs, contributor guide
- [ ] **Monitoring:** Sentry (errors) + simple analytics
- [ ] **Legal:** Privacy Policy, Terms of Service
- [ ] **Launch checklist:** SSL, domain, DNS, backup strategy

> **✅ DoD:** Aplikacja gotowa do pierwszych użytkowników. Dokumentacja kompletna.

---

## Podsumowanie Etapów

| Etap | Nazwa | Zależności | LOC est. | Sesji AI |
|------|-------|-----------|----------|----------|
| 0 | Scaffolding | — | ~200 | 1 |
| 1 | Shared Core | 0 | ~4000 | 3 |
| 2 | Health Module | 1 | ~3800 | 2-3 |
| 3 | Diet Module | 1 | ~2600 | 2 |
| 4 | Diagnosis Module | 1 | ~2600 | 2 |
| 5 | OCR Module | 1 | ~2800 | 2-3 |
| 6 | Visualization | 2,3,4 | ~3000 | 2-3 |
| 7 | Web Frontend | 1-6 | ~5000 | 3-4 |
| 8 | Mobile Frontend | 1-6 | ~4000 | 3-4 |
| 9 | MCP Servers | 1-6 | ~2400 | 2 |
| 10 | CI/CD | 0-9 | ~500 | 1 |
| 11 | Security | 1-10 | ~300 | 1 |
| 12 | Polish | 0-11 | ~800 | 2 |
| **TOTAL** | | | **~32,000** | **~25-30** |

---

## Reguły Dla AI Agenta

> Poniższe reguły MUSZĄ być przestrzegane przy każdym etapie.

### 🔴 Zasady bezwzględne:
1. **TDD:** Napisz test PRZED kodem. Kolejność: RED → GREEN → REFACTOR
2. **Zod:** Waliduj KAŻDE wejście/wyjście. Nigdy nie ufaj surowym danym
3. **RLS:** KAŻDA tabela z danymi użytkownika MUSI mieć Row-Level Security
4. **≤1500 LOC:** Jeśli moduł przekracza limit — podziel na sub-module
5. **Audit:** KAŻDY dostęp do danych medycznych jest logowany
6. **No secrets in code:** Wszystkie sekrety w `.env`, nigdy w kodzie
7. **GitHub Consent:** KAŻDY `push` wymaga uprzedniej zgody użytkownika i uzgodnienia brancha

### 🟡 Best practices:
1. **Hex Arch:** Core nie importuje frameworków (NestJS, Express, Drizzle)
2. **Ports:** Każda zewnętrzna zależność za interfejsem
3. **Naming:** `*.use-case.ts`, `*.port.ts`, `*.adapter.ts`, `*.entity.ts`
4. **Errors:** Custom error classes inheriting from `AppError`
5. **Logs:** Structured JSON logs (winston/pino)
6. **Comments:** Komentuj DLACZEGO, nie CO
