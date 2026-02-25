# User Stories - Projekt Zdrowie_v1

Dokument zawiera spis User Stories (historii użytkownika), które definiują funkcjonalność aplikacji z perspektywy końcowego odbiorcy.

## Etap 7: Frontend Web (Next.js)

### 1. Autentykacja i Bezpieczeństwo
- **US1: Logowanie**
  - *Jako:* Zarejestrowany użytkownik
  - *Chcę:* Móc bezpiecznie zalogować się do aplikacji (E-mail/Hasło)
  - *Aby:* Uzyskać dostęp do moich prywatnych danych zdrowotnych.
- **US2: Rejestracja**
  - *Jako:* Nowy użytkownik
  - *Chcę:* Móc założyć konto w aplikacji
  - *Aby:* Rozpocząć proces monitorowania zdrowia i diety.

### 2. Panel użytkownika (Dashboard)
- **US3: Dashboard Overview**
  - *Jako:* Użytkownik
  - *Chcę:* Widzieć podsumowanie moich statystyk (Health Score, ostatnia waga, spożyte kalorie) na jednej stronie
  - *Aby:* Szybko ocenić swój stan zdrowia bez zagłębinia się w szczegóły.
- **US4: Powiadomienia i Alerty**
  - *Jako:* Użytkownik
  - *Chcę:* Otrzymywać wizualne powiadomienia o anomaliach (np. wysokie tętno) lub przypomnienia
  - *Aby:* Móc szybko zareagować na potencjalne problemy.

### 3. Monitorowanie parametrów zdrowotnych
- **US5: Zarządzanie Wagą**
  - *Jako:* Użytkownik
  - *Chcę:* Regularnie wpisywać swoją wagę i widzieć trend na wykresie liniowym
  - *Aby:* Wiedzieć, czy zmierzam w stronę swojej docelowej masy ciała.
- **US6: Analiza Snu i Tętna**
  - *Jako:* Użytkownik
  - *Chcę:* Widzieć interaktywne wykresy moich faz snu oraz tętna spoczynkowego
  - *Aby:* Monitorować jakość regeneracji i wydolność organizmu.

### 5. Dieta i Odżywianie
- **US7: Logowanie Posiłków**
  - *Jako:* Użytkownik
  - *Chcę:* Dodawać zjedzone posiłki wraz z ich kalorycznością i makroskładnikami
  - *Aby:* Świadomie zarządzać swoją dietą i bilansem energetycznym.

### 6. Diagnostyka i AI
- **US8: Symptom Checker (Triage)**
  - *Jako:* Osoba odczuwająca dolegliwości
  - *Chcę:* Odpowiedzieć na zestaw pytań o objawy
  - *Aby:* Poznać poziom ryzyka i otrzymać rekomendację (np. wizyta u lekarza, SOR, odpoczynek).
- **US9: Cyfryzacja Dokumentów (OCR)**
  - *Jako:* Pacjent z wynikami papierowymi
  - *Chcę:* Załadować zdjęcie/skan wyników badań krwi
  - *Aby:* Dane automatycznie zasiliły mój profil zdrowotny bez ręcznego przepisywania.

### 7. Raporty i Eksport
- **US10: Generowanie Raportu PDF**
  - *Jako:* Pacjent przygotowujący się do wizyty u lekarza
  - *Chcę:* Wygenerować estetyczny raport PDF z moimi danymi i wykresami
  - *Aby:* Przedstawić lekarzowi kompletny obraz mojego stanu zdrowia z ostatniego okresu.

### 8. Personalizacja i Ustawienia (Onboarding & i18n)
- **US11: Kreator Powitalny (Onboarding)**
  - *Jako:* Nowo zarejestrowany użytkownik
  - *Chcę:* Przejść przez krokowy proces uzupełniania profilu (wiek, wzrost, początkowa waga, urządzenie, cele)
  - *Aby:* Aplikacja mogła lepiej dopasować dla mnie rekomendacje bez samodzielnego szukania ustawień.
- **US12: Internacjonalizacja (i18n)**
  - *Jako:* Użytkownik preferujący język angielski
  - *Chcę:* Zmienić język interfejsu aplikacji za pomocą zintegrowanego przełącznika
  - *Aby:* Swobodnie korzystać z systemu w języku, który jest dla mnie w pełni zrozumiały.

### 9. Integracje i Urządzenia
- **US13: Automatyczny odczyt parametrów z wagi (BLE)**
  - *Jako:* Użytkownik posiadający zewnętrzną wagę Bluetooth
  - *Chcę:* Zeskanować dostępne urządzenia, połączyć się z wagą za pomocą modułu BLE mobilnej aplikacji i pobrać aktualny wynik pomiaru masy
  - *Aby:* Oszczędzić czas i uniknąć literówek przy ręcznym wprowadzaniu danych.

### 10. Aspekty Prawne i Ochrona Danych
- **US14: Regulaminy i RODO (Privacy & Terms)**
  - *Jako:* Użytkownik dbający o bezpieczeństwo własnych danych wrażliwych (medycznych)
  - *Chcę:* Mieć klarowny podgląd Polityki Prywatności, zasad przetwarzania danych zgodnie z RODO i pełnego Regulaminu Usługi
  - *Aby:* Wiedzieć, komu i w jakim celu udostępniam historię parametrów życiowych.

---

*Uwaga: Powyższe historie są zintegrowane z testami E2E w `apps/web/tests/e2e/` (m.in.: uwierzytelnianie, onboarding, i18n, strony prawne) oraz z logiką domenową w odpowiednich modułach.*
