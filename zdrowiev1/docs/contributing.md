# Przewodnik dla Współtwórców (Contributing Guide)

Witaj! Dziękujemy za zaangażowanie w rozwój aplikacji **Zdrowie App**.

## Struktura Repozytorium (Turborepo)

Aplikacja zorganizowana jest w strukturze Monorepo:
- `apps/web/` - Portal użytkownika na Next.js
- `apps/api/` - Backend napisany w NestJS
- `apps/mobile/` - Aplikacja instalowana na telefon (React Native)
- `packages/` - Biblioteki i wspólne moduły

## TDD Workflow (Test-Driven Development)

Każda nowa klasa logiczna musi zostać poparta testami działającymi wg. Red-Green-Refactor. 

1. **RED:** Najpierw utwórz testy opisujące docelowe zachowanie. Komenda: `npx vitest` uruchomi je z informacją o błędzie.
2. **GREEN:** Napisz wystarczająco dużo kodu "produkcyjnego", by test zaświecił się na zielono.
3. **REFACTOR:** Usprawnij optymalizacyjnie kod, pilnując standardów.

## Nomenklatura Branchy (Branch Naming)

Używamy modyfikacji GitHub Flow z zachowaniem konwencyjnych prefixów:
- `feat/nazwa-funkcjonalnosci` - Główne funkcje, nowości
- `fix/nazwa-buga` - Usuwanie błędów
- `chore/aktualizacja-zaleznosci` - Techniczne aktualizacje
- `docs/aktualizacja-readme` - Prace dokumentacyjne

## Code Review Checklist

Zanim wystawisz Pull Request, zalecamy weryfikację:
- [ ] Zgodność formatowania (Biome).
- [ ] Czy kod logiczny ma min. 80% testów.
- [ ] Testy E2E nie raportują przerw (Playwright).
- [ ] Nie pozostawiono znaczników typu `console.log()` i `debugger`.
- [ ] Obsługa wyjątków nie pominęła bloków `catch`.
- [ ] Responsywność oraz obsługa i18n na nowych komponentach UI widokowych.

Dziękujemy za przestrzeganie standardów i współpracę!
