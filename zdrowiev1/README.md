# 🏥 Zdrowie App

Kompleksowa aplikacja do monitorowania zdrowia — waga, tętno, sen, dieta, diagnostyka (AI), OCR wyników badań. Zbudowana z myślą o najwyższej jakości (premium, dark-mode, glassmorphism) wg wzorców Hexagonal Modular Monolith.

## Key Features

- **Pulpit Główny (Dashboard)**: Health Score zasilany przez AI, widgeci, wykresy i powiadomienia zdrowotne.
- **Moduły**: Aktywność, Sen, Dieta, Diagnostyka i OCR wyników badań.
- **Integracje urządzeń**: Wspiera Garmin, Apple Health, Withings, Google Fit oraz skanowanie BLE w czasie rzeczywistym.
- **Bezpieczeństwo & Wersje językowe**: Pełne wsparcie HIPAA/GDPR, izolacja RLS w DB, oraz i18n (PL/EN).

## Tech Stack

- **Frontend/Web**: Next.js 15 (App Router), React, Tailwind v4, Framer Motion, next-intl
- **Backend**: NestJS, TypeScript
- **Baza Danych**: PostgreSQL + TimescaleDB (poprzez Prisma/TypeORM)
- **Aplikacja Mobilna**: React Native (Expo)
- **Narzędzia Jakościowe**: Turborepo, Vitest, Playwright, Biome

## Prerequisites

- Node.js 20+
- pnpm 8+ (jako menedżer pakietów monorepo)
- PostgreSQL 15+ (lub dostępny demon Docker)
- Docker Desktop (lokalne środowisko API/DB)

## Getting Started

### 1. Klonowanie repozytorium

```bash
git clone <repo-url>
cd zdrowiev1
```

### 2. Instalacja zależności

```bash
pnpm install
```

### 3. Konfiguracja zmiennych środowiskowych

Skopiuj przykładowe pliki `.env` (w katalogach `.env.example` jeśli istnieją):

| Zmienna | Opis | Przykład |
|---------|------|----------|
| `DATABASE_URL` | Adres bazy Postgres | `postgresql://user:pass@localhost:5432/zdrowie` |
| `JWT_SECRET` | Klucz prywatny tokenów | `super-secret-key-1234` |
| `SENTRY_DSN` | Monitorowanie błędów | `https://xxxx@o123.ingest.sentry.io/456` |
| `NEXT_PUBLIC_API_URL` | Adres API dla klienta | `http://localhost:3001` |

*(Pełna lista zmiennych znajduje się w dokumentacji konfiguracyjnej poszczególnych aplikacji `apps/api` i `apps/web`).*

### 4. Uruchomienie deweloperskie (Local)

Serwer Next.js (Web):
```bash
pnpm --filter web dev
```

Serwer NestJS (API z użyciem Dockera - patrz `docker-compose.yml`):
```bash
pnpm --filter api start:dev
```

*Frontend uruchomi się domyślnie na [http://localhost:3000](http://localhost:3000), a API na [http://localhost:3001](http://localhost:3001).*

## Architektura (Hexagonal Modular Monolith)

Architektura promuje ścisłą separację logiki biznesowej:
- `Domain` — modele, encje, czysty TypeScript.
- `Application` — przypadki użycia, TDD.
- `Infrastructure` — REST kontrolery, porty bazodanowe (Prisma).

## Dokumentacja wewnętrzna

| Dokument | Opis |
|----------|------|
| [api-docs.md](docs/api-docs.md) | Endpointy API z przykładami |
| [contributing.md](docs/contributing.md) | Standardy tworzenia PR, Git hooks, Testowanie |
| [architecture_analysis.md](docs/architecture_analysis.md) | Decyzje architektoniczne i bezpieczeństwo |
| [coding_standards.md](docs/coding_standards.md) | Wytyczne stylowe |

## Testing

Środowisko testowe jest ustawione w Vitest (Unit) oraz Playwright (E2E):

```bash
# Uruchom cały pakiet testów jednostkowych (API & Web)
pnpm test

# Testy E2E interfejsu klienta
pnpm --filter web test:e2e
```

## Podsumowanie Statusu
- **Faza**: Uruchamianie Produkcyjne (Launch) i testowanie E2E. Etap Web i Mobile (Etapy 7-12) jest zaawansowany z dodanymi modułami i18n, Onboardingu i Analizą wydajnościową (Lighthouse).

## Troubleshooting

- **Brak stylów Tailwind z klas `:local()`**: Upewnij się, że nie wywołujesz niepoprawnych prefixów.
- **Porty w użyciu (3000/3001)**: Użyj komend `kill` dla procesów blokujących lub zmień port w `.env`.
- **E2E Playwright Timeout**: Uruchom środowisko powoli lub sprawdź zmienną bazową `baseURL` na porcie 3000.
