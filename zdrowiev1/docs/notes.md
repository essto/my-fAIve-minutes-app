# Notatki z Projektu

Ten plik będzie służył do gromadzenia kluczowych informacji, decyzji projektowych oraz wyników researchu w trakcie naszej pracy. Na końcu zostanie on skonsolidowany w formie końcowego raportu lub podsumowania.

## 📝 Bieżące Notatki
- [2026-02-21] Analiza dostępnych skilli zakończona (frontend, testing, research).
- [2026-02-21] Weryfikacja OpenRouter (tokeny: ~37k, balans: ~$8.76).
- [2026-02-21] Inicjalizacja pliku z notatkami.

## 📌 Kluczowe Decyzje
*(Będą dopisywane w trakcie prac)*

## 🏥 Projekt: Aplikacja Monitorowania Zdrowia

### Research Konkurencji
- **Withings Health Mate**: Świetna integracja z wagami, ale skupiona głównie na ekosystemie Withings.
- **Garmin Connect**: Bardzo szczegółowe dane sportowe, ale interfejs bywa przytłaczający.
- **Fitbit**: Dobry dla początkujących, silny aspekt społecznościowy.
- **Apple Health / Google Fit**: Agregatory danych, brak dedykowanych aplikacji desktopowych o pełnej funkcjonalności.

### Lista Funkcjonalności (MVP)
1. **Pulpit nawigacyjny (Dashboard)**: Czytelny podgląd wagi, tętna, snu i kroków.
2. **Integracje**: Synchronizacja z Apple HealthKit, Google Fit oraz bezpośrednio z wagami (WiFi/Bluetooth).
3. **Analiza Trendów**: Wykresy zmian wagi i składu ciała w czasie.
4. **Desktop Sync**: Pełen dostęp do danych na komputerze (wersja stacjonarna).
5. **Powiadomienia**: Przypomnienia o pomiarach, cele dzienne.

## 📋 Utylizacja Wymagań (Requirements Gathering)

### Wymagania Funkcjonalne (Co aplikacja ma robić?)
1. **Zarządzanie Profilem**: Wiek, wzrost, płeć, cele zdrowotne.
2. **Monitoring Wagi**:
    - Automatyczny zapis z wag smart (WiFi/BT).
    - Ręczne wprowadzanie danych.
    - śledzenie składu ciała (BMI, tłuszcz, mięśnie, woda).
3. **Monitoring Aktywności (Zegarki)**:
    - Synchronizacja kroków, tętna i spalonych kalorii.
    - Monitorowanie jakości i faz snu.
    - Automatyczne wykrywanie treningów (jeśli pozwala API).
4. **Wersja Desktopowa**:
    - Zaawansowane raporty miesięczne/roczne.
    - Możliwość eksportu danych (PDF/CSV).
    - Porównywanie wyników z różnych okresów.
5. **Wersja Mobilna**:
    - Tryb "Szybki rzut oka" na bieżący dzień.
    - Powiadomienia push (przypomnienia, gratulacje).

### Wymagania Niefunkcjonalne (Jak aplikacja ma działać?)
- **Synchronizacja**: Dane muszą być spójne między telefonem a desktopem w czasie rzeczywistym.
- **Dostępność offline**: Możliwość przeglądania ostatnich danych bez internetu.
- **Prywatność**: Szyfrowanie danych medycznych i wagi na urządzeniu i w chmurze.

## 📊 Analiza Porównawcza Konkurencji

Poniżej zestawienie najlepszych, unikalnych funkcjonalności od liderów rynku w 5 kluczowych kategoriach.

### 1 & 2. Monitorowanie Zdrowia i Hardware (Zegarki/Wagi)

#### [Garmin Connect](https://www.garmin.com/en-US/p/125677)
- **Body Battery**: Monitorowanie rezerw energii organizmu w czasie rzeczywistym.
- **Health Snapshot**: 2-minutowa sesja rejestrująca kluczowe statystyki (HR, HRV, SpO2).
- **Traning Readiness**: Ocena gotowości do treningu na podstawie obciążenia i snu.
- **Incident Detection**: Automatyczne powiadamianie o wypadkach/upadkach.
- **LiveTrack**: Udostępnianie lokalizacji i wyników bliskim w czasie rzeczywistym.
- **Garmin Coach**: Adaptacyjne plany treningowe pod okiem ekspertów.
- **Menstrual Cycle & Pregnancy Tracking**: Śledzenie cyklu z poradami dotyczącymi treningu i diety.
- **ClimbPro**: Analiza podjazdów i wzniesień w czasie rzeczywistym (dla rowerzystów/biegaczy).
- **VO2 Max for Trail Run**: Precyzyjne szacowanie wydolności w trudnym terenie.
- **Advanced Sleep Insights**: Deep dive w fazy snu z punktacją i sugestiami.

#### [Apple Health](https://www.apple.com/ios/health/)
- **Activity Rings**: Kultowy system zamykania pierścieni (Ruch, Ćwiczenie, Na nogach).
- **Health Records**: Integracja z systemami szpitalnymi i pobieranie wyników badań.
- **Hearing Health**: Monitorowanie poziomu hałasu otoczenia i ochrona słuchu.
- **Walking Steadiness**: Analiza stabilności chodu w celu zapobiegania upadkom.
- **Cycle Tracking with Wrist Temp**: Wykrywanie owulacji na podstawie temperatury z Apple Watch.
- **Medications Tracking**: Przypomnienia o lekach z logowaniem interakcji.
- **Emergency SOS & Fall Detection**: Najwyższy standard wykrywania upadków i alarmowania służb.
- **ECG & Irregular Rhythm Notifications**: Certyfikowane medycznie wykrywanie migotania przedsionków.
- **Mental Health Logging**: Śledzenie nastroju i ocena ryzyka depresji/lęku.
- **Health Sharing**: Bezpieczne udostępnianie wybranych danych rodzinie lub lekarzowi.

#### [Withings Health Mate](https://www.withings.com/us/en/health-mate)
- **Vascular Age**: Pomiar sztywności tętnic na podstawie prędkości fali tętna.
- **Nerve Activity Score**: Ocena zdrowia nerwów poprzez potliwość stóp na wagach.
- **Segmental Body Composition**: Precyzyjny pomiar tłuszczu/mięśni osobno dla rąk, nóg i tułowia.
- **Sleep Apnea Detection**: Wykrywanie bezdechu sennego bez noszenia urządzeń (mata pod materac).
- **Trend Analysis**: Bardzo czytelne wykresy "smoothing" dla wagi, eliminujące dzienne wahania.
- **Automatic Multi-user Recognition**: Rozpoznawanie do 8 użytkowników na wagach bez klikania.
- **Pregnancy Mode**: Śledzenie wagi w ciąży z instrukcjami od położnych.
- **Eyes-closed Mode**: Ważenie się bez pokazywania liczby (tylko trend), by uniknąć stresu.
- **Baby Mode**: Ważenie dziecka przez wejście na wagę z nim na rękach.
- **Integration with 100+ Apps**: Jeden z najlepiej połączonych ekosystemów na rynku.

### 3 & 4. Dieta i Zdrowe Odżywianie

#### [MyFitnessPal](https://www.myfitnesspal.com/)
- **Huge Food Database**: Ponad 14 mln produktów z globalną bazą UPC.
- **Recipe Importer**: Automatyczne obliczanie makro z dowolnego linku do przepisu.
- **AI Meal Scan**: Rozpoznawanie jedzenia na talerzu na podstawie zdjęcia.
- **Macro Goals by Meal**: Ustawianie różnych celów makro dla śniadania, obiadu itp.
- **Community Transformation**: Jedna z największych baz sukcesów i wsparcia społeczności.
- **Barcode Scanner**: Błyskawiczne logowanie produktów gotowych.
- **Exercise Calorie Adjustment**: Automatyczne odejmowanie spalonych kalorii od limitu jedzenia.
- **Custom Food Creation**: Ładne dodawanie własnych dań "domowych".

#### [Cronometer](https://cronometer.com/)
- **Micronutrient Tracking**: Najdokładniejszy monitoring 82 witamin i minerałów.
- **Oracle (Food Search)**: Podpowiadanie produktów o najwyższej gęstości odżywczej.
- **Laboratory-grade Data**: Weryfikacja każdego produktu przez zespół dietetyków.
- **Full Integration with Wearables**: Najlepsza synchronizacja z Garmin/Oura/Apple Watch.
- **Fasting Timer**: Zintegrowany timer postu przerywanego (IF).
- **Bio-marker Tracking**: Logowanie wyników krwi, temperatury, glukozy.
- **Nutrient Targets**: Personalizowane cele dla specyficznych diet (np. Keto, Weganizm).

#### [Yuka](https://yuka.io/en/)
- **Health Score**: Prosta ocena (0-100) na podstawie składu, dodatków i certyfikatów.
- **Harmful Additives Detection**: Wykrywanie kontrowersyjnych E-dodatków.
- **Healthier Alternatives**: Sugerowanie lepszych zamienników dla "złych" produktów.
- **Sustainability Rating**: Ocena wpływu produktu na środowisko (Eco-score).
- **Cosmetic Scanning**: Analiza składu kosmetyków i chemii domowej.
- **Historical Analysis**: Statystyka jakości zakupów z całego miesiąca.

### 5. Diagnostyka i Objawy (AI Diagnosis)

#### [Ada Health](https://ada.com/)
- **Medical Dictionary AI**: Analiza objawów w oparciu o tysiące schorzeń.
- **Dynamic Assessment**: Pytania zmieniające się w zależności od Twoich odpowiedzi (jak u lekarza).
- **Symptom Tracker**: Śledzenie ewolucji bólu lub objawów w czasie.
- **Condition Library**: Bardzo obszerne opisy chorób przygotowane przez lekarzy.
- **Assessment Reports**: Generowanie PDF dla lekarza z historią symptomów.
- **Risk Factor Assessment**: Uwzględnianie genetyki, stylu życia i podróży w diagnozie.

#### [Symptomate](https://symptomate.com/)
- **Triage Recommendation**: Jasne wytyczne: "Zostań w domu", "Zadzwoń do lekarza", "SOR".
- **Multi-user Interviews**: Osobne profile dla dzieci i dorosłych (różne algorytmy).
- **Body Map Interface**: Wskazywanie bólów bezpośrednio na modelu 3D ciała.
- **Health Checkup Mode**: Rutynowe przeglądy zdrowia wykonywane co jakiś czas.
- **Doctor Consultation Integration**: Bezpośrednie połączenie z telemedycyną po diagnozie.
