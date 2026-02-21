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

## Sesja 4: Przygotowanie do GitHub

### Prompt 9: Historia + push na GitHub
> *"Zapisz historię w app_history.md, przygotuj wszystko do wysłania na GitHub."*

**Co zrobiono:**
- Utworzono `app_history.md` (ten plik)
- Skopiowano wszystkie dokumenty do projektu `Zdrowie_v1/docs/`
- Przygotowano do git push

---

## Wytworzone Dokumenty

| Plik | Opis | Rozmiar |
|------|------|---------|
| `architecture_analysis.md` | Analiza 3 architektur, stack, auth, RLS, Zod, testy, DB schema | ~730 linii |
| `detailed_plan.md` | 13 etapów implementacji z krokami dla AI | ~500 linii |
| `notes.md` | Research konkurencji, wymagania, lista features | ~140 linii |
| `app_history.md` | Historia sesji (ten plik) | ~130 linii |

## Kluczowe Decyzje

1. **Architektura:** Hexagonal Modular Monolith (nie mikroserwisy — za wcześnie)
2. **Frontend:** Next.js 15 (web) + React Native Expo (mobile)
3. **Backend:** NestJS + TypeScript + Drizzle ORM
4. **Baza:** PostgreSQL + TimescaleDB + Row-Level Security
5. **AI Integration:** MCP Servers wrapping REST API
6. **Walidacja:** Zod (single source of truth for types)
7. **Testy:** TDD obowiązkowe, 5 typów testów, pokrycie core ≥90%
8. **OCR:** Tesseract.js + LLM z anonimizacją danych
9. **Wizualizacje:** Recharts (web) + Victory Native (mobile), 10 typów wykresów
10. **Bezpieczeństwo:** 4-warstwowa izolacja, HIPAA/GDPR, Consent Management
