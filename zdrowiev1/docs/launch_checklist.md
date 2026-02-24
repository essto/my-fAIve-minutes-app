# Zdrowie App - Launch Checklist

Poniższa lista kontrolna służy jako przewodnik przed uruchomieniem wersji produkcyjnej aplikacji Zdrowie App. Należy upewnić się, że wszystkie kroki zostały spełnione.

## 1. Weryfikacja SSL i Domen

- [ ] Certyfikaty SSL podpięte do domen (Frontend: `zdrowieapp.pl` / Backend: `api.zdrowieapp.pl`).
- [ ] Automatyczne przekierowanie HTTP -> HTTPS włączone.
- [ ] Skonfigurowane rekordy DNS (A, AAAA, CNAME dla IPv6) z uwzględnieniem usług pocztowych (MX, TXT, SPF, DKIM, DMARC) w razie wysyłania e-maili.

## 2. Walidacja Środowiska Produkcyjnego (Production Environment Validation)

- [ ] **Zmienne Środowiskowe (ENV)**:
  - Skonfigurowane zmienne produkcyjne (brak `localhost`).
  - Wyłączone tryby `DEBUG`.
  - Prawidłowy `NODE_ENV=production`.
  - Wymienione klucze JWT i Secret Key na generowane losowo (min 64 znaki).
  - Integracje zewnętrznych API (Garmin, Apple) korzystają z kluczy produkcyjnych.
- [ ] **Baza Danych**:
  - Działające repliki i włączony failover (High Availability).
  - Restrykcje sieciowe (VPC, firewall) ograniczające dostęp z zewnątrz wyłącznie do backendu NestJS.
  - Włączony Row-Level Security (RLS) w Postgre (lub odpowiedniki) zapobiegający dostępowi cross-tenantowym.

## 3. Strategia Kopii Zapasowych (Backup Strategy Documentation)

- [ ] **Bazy Danych (PostgreSQL + TimescaleDB)**:
  - Zautomatyzowane snapshoty wykonywane codziennie (częstotliwość max. co 24h w godzinach nocnych tj. 02:00:00).
  - Włączone Continuous Archiving (np. WAL-G/pgBackRest) z PITR (Point-in-Time Recovery) pozwalające cofnąć stan bazy danych max do 7 dni wstecz z granularnością sekundową.
- [ ] **Nośniki**: Kopia trzymana w chmurze obiektu (S3, GCS) z oddzielnego dostawcy dla zachowania redundancji. Retencja min. 30 dni.

## 4. Monitoring i Alerty

- [ ] Sentry (lub Bugsnag) otrzymuje eventy z włączonymi mapami błędów (Source Maps w next).
- [ ] Wpięty monitoring metryk infrastruktury (np. Datadog, Prometheus+Grafana) (CPU, RAM, DB Connections).
- [ ] Alerty na Slack/Email dla P99 Latency > 1000ms, Error Rate 500 > 1%.

## 5. Formalności Prawne

- [ ] Polityka prywatności opisuje metody obróbki dla modeli AI (szczególnie anonimizację przed requestem do OpenAI/Anthropic).
- [ ] Potwierdzona gotowość na usuwanie kont i danych telemetrycznych na wniosek RODO.
