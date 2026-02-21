# 📜 Historia Projektu: Zdrowie App

> **Data sesji:** 2026-02-21
> **Cel:** Zaplanować od zera aplikację monitorowania zdrowia z architekturą gotową do budowy przez AI agenta.

---

## Sesja 1: Eksploracja Narzędzi i Research

### Prompt 1: Odkrywanie skilli
> *"Jakie masz skille? Sprawdź co masz pod ręką."*

**Co zrobiono:**
- Przeskanowano 20+ skilli Antigravity (frontend-design, webapp-testing, firecrawl, mcp-builder, etc.)
- Sprawdzono stan OpenRouter: ~37k tokenów, ~$8.76 balansu
- Utworzono `skills_overview.md` z opisem dostępnych narzędzi

### Prompt 2: Inicjalizacja projektu zdrowotnego
> *"Chcę stworzyć aplikację zdrowotną — monitoring wagi, tętna, snu, integracja z zegarkami i wagami."*

**Co zrobiono:**
- Research konkurencji: Withings, Garmin, Fitbit, Apple Health, Google Fit
- Identyfikacja nisz (brak wersji desktopowych, zamknięte ekosystemy)
- Wstępna lista funkcjonalności MVP
- Pierwsza propozycja technologii: Flutter
- Utworzono `implementation_plan.md` (pierwsza wersja) i `notes.md`

### Prompt 3: Korekta kursu — zbieranie wymagań
> *"Za wcześnie na plan, narazie zbieramy wymagania. Ty zapisuj."*

**Co zrobiono:**
- Przesunięcie fokusa z planowania na Requirements Gathering
- Dodanie sekcji wymagań funkcjonalnych i niefunkcjonalnych do `notes.md`
- Pytania do usera: wielu użytkowników? import CSV? raporty dla lekarza? konkretne urządzenia?

**Odpowiedź usera:** Na wszystkie 4 pytania — TAK.

---

## Sesja 2: Deep Research Konkurencji

### Prompt 4: Głęboka analiza konkurencji w 5 kategoriach
> *"Zrób listę najlepszych apps w kategoriach: zdrowotne, zegarki/wagi, dietetyczne, zdrowe odżywianie, diagnozowanie chorób. Dla każdej po 10-15 najlepszych funkcjonalności."*

**Co zrobiono:**
- **Kategoria 1 & 2 (Zdrowie + Hardware):**
  - Garmin Connect (10 features: Body Battery, Health Snapshot, Training Readiness...)
  - Apple Health (10 features: Activity Rings, Health Records, ECG, Mental Health...)
  - Withings Health Mate (10 features: Vascular Age, Sleep Apnea, Segmental Composition...)

- **Kategoria 3 & 4 (Dieta + Odżywianie):**
  - MyFitnessPal (8 features: 14M food database, AI Meal Scan, Recipe Importer...)
  - Cronometer (7 features: 82 micronutrients, Oracle search, Fasting Timer...)
  - Yuka (6 features: Health Score, Harmful Additives, Eco-score...)

- **Kategoria 5 (Diagnostyka AI):**
  - Ada Health (6 features: Dynamic Assessment, Symptom Tracker, Risk Factors...)
  - Symptomate (5 features: Triage, Body Map 3D, Health Checkup Mode...)

- Wszystkie zapisane w `notes.md`

---

## Sesja 3: Architektura i Technologie

### Prompt 5: Wybór architektury
> *"Rozważ microservices, modular monolith, hexagonal. Chcę: przeglądarkę i mobile, AI MCP maintenance, niezależną grafikę, bezpieczeństwo danych medycznych, łatwą integrację urządzeń, TDD."*

**Co zrobiono:**
- Deep research: 7 wyszukiwań + 3 artykuły techniczne
- Porównanie 3 architektur z tabelami za/przeciw
- **Rekomendacja: Hexagonal Modular Monolith**
- API-Centric MCP Strategy (REST + MCP wrappers)
- Frontend: Next.js (web) + React Native (mobile) — bo PWA nie ma BLE na iOS
- Bezpieczeństwo HIPAA/GDPR
- Stack: NestJS, PostgreSQL+TimescaleDB, Drizzle, Redis
- Utworzono `architecture_analysis.md`

### Prompt 6: Wyjaśnienie rekomendacji
> *"Wytłumacz dlaczego hexagonal modular i jak to ma wyglądać?"*

**Co zrobiono:**
- Szczegółowe wyjaśnienie z przykładami kodu (port, adapter, use case, test)
- Diagram izolacji warstw
- Porównanie kosztów vs mikroserwisy
- Mapowanie wymagań usera → rozwiązania w Hex Arch

### Prompt 7: Uzupełnienie o brakujące elementy
> *"Dodaj: Zod bazy, testy per moduł, bezpieczeństwo, autoryzację, izolację danych użytkowników."*

**Co zrobiono:**
- **§8 Zod Schemas** — pełne schematy dla: User, Consent, Weight, HeartRate, Sleep, Meal, Symptom
- **§9 Testy** — matryca 5 typów: unit/integration/contract/E2E/security z progami pokrycia
- **§10 Autoryzacja** — JWT + RBAC (4 role) + ABAC + ConsentGuard
- **§11 Izolacja** — 4-warstwowa: JWT Guard → ConsentGuard → Repository → PostgreSQL RLS
- **§12 Schemat DB** — ER diagram: Users, Consents, WeightReadings, AuditLogs, DeviceConnections
- Zaktualizowano `architecture_analysis.md`

### Prompt 8: Kompletny plan + OCR + Wizualizacje
> *"Dodaj wizualizację danych, OCR dla skanowania dokumentów. Zrób szczegółowy plan w osobnym pliku, podzielony na etapy, moduły ≤1500 LOC, plan dla AI."*

**Co zrobiono:**
- **OCR Pipeline:** Tesseract.js (local) + LLM (OpenRouter) + anonimizacja + user verification
- **10 typów wykresów:** line, area, bar, radar, gauge, heatmap, scatter, progress rings, sparkline, candlestick
- **13 etapów** od scaffolding do launch
- **30+ modułów** — każdy ≤1500 LOC z hex arch
- **Reguły dla AI** — TDD obowiązkowe, Zod na każdym wejściu, RLS na każdej tabeli
- Szacunek: ~32k LOC, ~25-30 sesji AI
- Utworzono `detailed_plan.md`

---

## Sesja 5: Standardy Jakości i Polityka Kodu

### Prompt 10: Dobre praktyki i automatyzacja
> *"czy mamy jekies dobre praktyki robienia takiego wew api... czy mamy husky i inliner... kto bedize pilnował czy kod jest zoptymalizwany?"*

**Co zrobiono:**
- Utworzono **`coding_standards.md`** — nadrzędny dokument zasad (konstytucja projektu).
- Zdefiniowano komunikację wewnętrzną przez Porty/DI (opóźnienie 0ms).
- Zaplanowano setup **Husky + lint-staged** dla blokowania złego kodu.
- Wprowadzono limity: max 50 linii na funkcję, max 1500 na moduł.
- Ustalono rolę AI Agenta jako "Boy Scout" (zawsze czyściej po pracy).

---

## Sesja 6: Testy, Serwery i Smoke Tests

### Prompt 11: Standardy testowania i procedury serwerowe
> *"dodaj standardy testownia oraz procedury uruchamiania serverów... mamy teswt unit, integracyjne e2e itd... mamy smoke tsty?"*

**Co zrobiono:**
- Rozbudowano hierarchię testów: Unit, Integration, Contract, E2E.
- Dodano **Smoke Tests** (§3.2) dla błyskawicznej weryfikacji po deployu.
- Ustalono standardową Mapę Portów (5432, 6379, 3000, 4000, 8080).
- Zdefiniowano kolejność startu: Infra → API → Web.

---

## Sesja 7: AI Readiness (pgvector)

### Prompt 12: Bazy wektorowe
> *"czy nasza architektura jest gotowa na baze danych ai... rpzechowuje wyrazy jako liczby?"*

**Co zrobiono:**
- Analiza gotowości na AI: wybór **`pgvector`** dla PostgreSQL.
- Zdefiniowano RAG (Retrieval-Augmented Generation) dla asystenta zdrowia.
- Zapewniono bezpieczeństwo wektorów przez Row-Level Security (RLS).
- Dodano §13 do `architecture_analysis.md`.

---

## Sesja 8: Warstwa MCP (AI-Native API)

### Prompt 13: Szczegóły MCP
> *"warstaa mcp na czym polega i jak bedzie działała i na jakim etapie ja dodamy?"*

**Co zrobiono:**
- Wyjaśniono MCP jako interfejs dla maszyn/agentów.
- Potwierdzono wdrożenie w **Etapie 9** (po zbudowaniu Core).
- Zdefiniowano zastosowania: profilaktyka, asystent, autonomous maintenance.

---

## Wytworzone Dokumenty

| Plik | Opis | Rozmiar |
|------|------|---------|
| `architecture_analysis.md` | Analiza 3 architektur, stack, auth, RLS, Zod, testy, DB schema, **AI/Vector**, **MCP** | ~750 linii |
| `detailed_plan.md` | 13 etapów implementacji z krokami dla AI, moduły ≤1500 LOC | ~680 linii |
| `coding_standards.md` | **NOWY:** Standardy kodowania, TDD, Hex Arch, Clean Code | ~120 linii |
| `notes.md` | Research konkurencji, wymagania, lista features | ~140 linii |
| `app_history.md` | Historia sesji (ten plik) | ~200 linii |

## Kluczowe Decyzje

1. **Architektura:** Hexagonal Modular Monolith.
2. **Frontend:** Next.js 15 (web) + React Native Expo (mobile).
3. **Backend:** NestJS + TypeScript + Drizzle ORM.
4. **Baza:** PostgreSQL + TimescaleDB + **pgvector** (dla AI) + Row-Level Security.
5. **AI Integration:** Hybryda REST API + **MCP Servers** (Etap 9).
6. **Walidacja:** Zod (single source of truth).
7. **Testowanie:** TDD, 5 typów testów (**w tym Smoke Tests**).
8. **Automatyzacja:** Husky, ESLint (limit 50 linii/funkcja), CI/CD coverage 90%.
9. **OCR:** Tesseract.js (local) + LLM (OpenRouter) z anonimizacją.
10. **Wizualizacje:** Recharts + Victory Native, 10 typów wykresów.
