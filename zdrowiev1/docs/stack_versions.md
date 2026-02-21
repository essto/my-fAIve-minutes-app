# 📦 Zestawienie Wersji Technologicznych (LTS & Locked)

> Dokument ten definiuje "Złoty Standard" wersji dla projektu Zdrowie App. Zmiana dowolnej z tych wersji wymaga uzgodnienia i aktualizacji w całym monorepo.

## 1. Fundamenty Runtine
| Element | Wersja | Uwagi |
|---------|--------|-------|
| **Node.js** | `v22.x (LTS)` | Najnowszy LTS (Jod) |
| **npm** | `v10.x` | Dołączony do Node v22 |
| **TypeScript** | `v5.7.3` | Ścisła typizacja (Strict mode) |
| **Docker** | `v24+` | Wsparcie dla Compose V2 |

## 2. Backend (NestJS Stack)
| Biblioteka | Wersja | Cel |
|------------|--------|-----|
| **NestJS** | `v11.x` | Główny framework backendowy |
| **Drizzle ORM** | `v0.39.5` | Lekki ORM SQL-first |
| **drizzle-kit** | `v0.30.4` | Narzędzia do migracji |
| **Zod** | `v3.24.1` | Walidacja runtime |
| **PostgreSQL** | `v16.x` | Baza z wsparciem pgvector |
| **Redis** | `v7.2` | Cache i sesje |

## 3. Frontend (Web & Mobile)
| Biblioteka | Wersja | Cel |
|------------|--------|-----|
| **Next.js** | `v15.1.6` | App Router, React 19 compatibility |
| **React** | `v19.0.0` | Stabilna wersja 19 |
| **Expo (React Native)**| `v52.x` | Stabilne SDK dla Mobile |
| **Tailwind CSS** | `v3.4.x` | Ostylowanie (jeśli wymagane) |
| **Framer Motion** | `v11.x` | Animacje Premium |
| **TanStack Query** | `v5.x` | Zarządzanie stanem asynchronicznym |

## 4. Narzędzia Deweloperskie & QA
| Biblioteka | Wersja | Cel |
|------------|--------|-----|
| **Turbo (Turborepo)** | `v2.x` | Zarządzanie monorepo |
| **Vitest** | `v3.x` | Szybkie testy Unit/Integration |
| **Playwright** | `v1.50.x` | Testy E2E Web |
| **Husky** | `v9.x` | Pre-commit hooks |
| **Commitlint** | `v19.x` | Spójność wiadomości git |

---

## 🔒 Reguła Blokady (Version Pinning)
1. **package.json:** Używamy dokładnych wersji (bez `^` ani `~`), np. `"drizzle-orm": "0.39.5"`.
2. **Engines:** W głównym `package.json` ustawiamy `"engines": { "node": ">=22.0.0" }`.
3. **CI/CD:** GitHub Actions zawsze używa wersji zdefiniowanych powyżej.
