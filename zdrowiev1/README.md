# 🏥 Zdrowie App

Kompleksowa aplikacja monitorowania zdrowia — waga, tętno, sen, dieta, diagnostyka, OCR wyników badań.

## 🏗️ Architektura

**Hexagonal Modular Monolith** — połączenie elastyczności heksagonalnej z pragmatyzmem monolitu.

| Warstwa | Technologia |
|---------|-------------|
| Backend | NestJS + TypeScript |
| Baza danych | PostgreSQL + TimescaleDB |
| Web | Next.js 15 |
| Mobile | React Native (Expo) |
| AI Integration | Custom MCP Servers |
| Testy | Vitest + Playwright |

## 📂 Dokumentacja

| Dokument | Opis |
|----------|------|
| [architecture_analysis.md](docs/architecture_analysis.md) | Analiza architektur, stack, auth, bezpieczeństwo, schemat DB |
| [detailed_plan.md](docs/detailed_plan.md) | 13-etapowy plan implementacji (krok po kroku dla AI) |
| [coding_standards.md](docs/coding_standards.md) | Standardy kodowania, TDD, Hex Arch, Clean Code |
| [notes.md](docs/notes.md) | Research konkurencji, wymagania, lista features |
| [app_history.md](docs/app_history.md) | Historia sesji planowania |

## 🎯 Kluczowe Funkcje

- **Dashboard** — Health Score, Activity Rings, trendy
- **Monitoring** — waga, tętno, sen, aktywność
- **Dieta** — logowanie posiłków, barcode scanner, makroskładniki
- **Diagnostyka** — checker objawów, triage AI, raporty PDF
- **OCR** — skanowanie wyników badań, AI interpretacja
- **Wizualizacje** — 10 typów wykresów, eksport PDF/PNG
- **Integracje** — Garmin, Apple Health, Withings, Google Fit, BLE
- **Bezpieczeństwo** — HIPAA/GDPR, RLS, 4-warstwowa izolacja danych

## 🚀 Status

📋 **Faza:** Implementacja Etapu 7 (Frontend Web) w toku → Większość modułów Core gotowa do integracji mobilnej (Etap 8)

## 📜 Licencja

Projekt prywatny.
localStorage.setItem('token', 'demo-token')
location.href = '/dashboard'
