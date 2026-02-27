# 📱 Przewodnik: Konfiguracja i Uruchomienie Aplikacji Mobilnej "Zdrowie"

> **Cel:** Ten dokument prowadzi Cię krok po kroku przez konfigurację Android Studio, emulatora, Maestro Studio i Expo, abyś mógł uruchomić i przetestować aplikację mobilną.

---

## Wymagane Programy (Zainstalowane ✅)

| Program | Lokalizacja | Status |
|---------|------------|--------|
| **Android Studio** | `C:\Program Files\Android\Android Studio` | ✅ Zainstalowany |
| **Maestro Studio** | `C:\Users\pszymelyniec\AppData\Local\Programs\Maestro Studio` | ✅ Zainstalowany |
| **Node.js + npm** | Systemowy PATH | ✅ Dostępny (monorepo działa) |

---

## ETAP 1: Konfiguracja Android Studio i SDK

> [!IMPORTANT]  
> Android SDK **nie został jeszcze pobrany**. Musisz to zrobić przy pierwszym uruchomieniu Android Studio.

### Krok 1.1: Uruchom Android Studio
1. Otwórz menu Start → wpisz **"Android Studio"** → kliknij
2. Przy pierwszym uruchomieniu pojawi się **Setup Wizard**:
   - Wybierz **Standard** installation
   - Zaakceptuj licencje (kliknij "Accept" przy każdej)
   - Kliknij **Finish** — rozpocznie się pobieranie SDK (~2-3 GB)
3. **Poczekaj** aż pobieranie się zakończy (może trwać 10-30 min)

### Krok 1.2: Stwórz emulator (Virtual Device)
1. Na ekranie powitalnym Android Studio kliknij **More Actions** → **Virtual Device Manager**
2. Kliknij **Create Virtual Device**
3. Wybierz urządzenie: **Pixel 8** → kliknij **Next**
4. Pobierz obraz systemu: przy **API 34 (UpsideDownCake)** kliknij **Download** → poczekaj → **Next**
5. Zostaw domyślne ustawienia → **Finish**
6. Na liście urządzeń kliknij ▶️ **Play** przy swoim Pixel 8 → emulator się uruchomi

### Krok 1.3: Ustaw zmienne środowiskowe
Po zakończeniu Setup Wizard, SDK powinno być w:
```
C:\Users\pszymelyniec\AppData\Local\Android\Sdk
```

Otwórz **PowerShell jako Administrator** i wklej:

```powershell
# Ustaw ANDROID_HOME na stałe (User level)
[System.Environment]::SetEnvironmentVariable("ANDROID_HOME", "$env:LOCALAPPDATA\Android\Sdk", "User")

# Ustaw JAVA_HOME na JDK wbudowaną w Android Studio
[System.Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Android\Android Studio\jbr", "User")

# Dodaj narzędzia do PATH
$currentPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$sdkPath = "$env:LOCALAPPDATA\Android\Sdk"
$newPaths = "$sdkPath\emulator;$sdkPath\platform-tools;$sdkPath\cmdline-tools\latest\bin"
if ($currentPath -notlike "*$sdkPath*") {
    [System.Environment]::SetEnvironmentVariable("Path", "$currentPath;$newPaths", "User")
}

Write-Host "✅ Zmienne ustawione. Zamknij i otwórz NOWY terminal aby zadziałały."
```

### Krok 1.4: Weryfikacja (w NOWYM terminalu PowerShell)
```powershell
echo $env:ANDROID_HOME    # Powinno wyświetlić: C:\Users\pszymelyniec\AppData\Local\Android\Sdk
adb devices               # Powinno wyświetlić listę urządzeń (jeśli emulator działa)
emulator -list-avds       # Powinno wyświetlić nazwę Twojego AVD (np. "Pixel_8_API_34")
```

---

## ETAP 2: Uruchomienie Emulatora Android

### Metoda A: Z Android Studio (najprostsza)
1. Otwórz **Android Studio**
2. Kliknij **More Actions** → **Virtual Device Manager**
3. Kliknij ▶️ **Play** przy swoim urządzeniu

### Metoda B: Z terminala (szybsza po konfiguracji)
```powershell
# Najpierw sprawdź nazwę AVD:
emulator -list-avds

# Uruchom emulator (zamień nazwę na swoją):
emulator -avd Pixel_8_API_34

# Weryfikacja że emulator działa:
adb devices
# Powinno wyświetlić coś jak: emulator-5554  device
```

---

## ETAP 3: Uruchomienie Aplikacji Expo na Emulatorze

### Krok 3.1: Zainstaluj zależności mobilne
```powershell
cd c:\od_zera_do_ai\Zdrowie_v1\zdrowiev1
npm install       # jeśli jeszcze nie było robione

cd apps/mobile
npx expo install  # synchronizuj wersje Expo
```

### Krok 3.2: Uruchom Expo Dev Server
```powershell
cd c:\od_zera_do_ai\Zdrowie_v1\zdrowiev1\apps\mobile
npx expo start
```

W terminalu pojawi się kod QR i menu opcji:
```
› Press a │ open Android
› Press w │ open web
› Press r │ reload app
› Press m │ toggle menu
› Press j │ open debugger
```

### Krok 3.3: Wyślij aplikację na emulator
- Upewnij się że **emulator Android działa** (krok ETAP 2)
- Naciśnij klawisz **`a`** w terminalu Expo
- Expo automatycznie zainstaluje i uruchomi aplikację na emulatorze
- Za pierwszym razem może to potrwać 2-3 minuty (buduje paczkę)

> [!TIP]
> Jeśli `a` nie działa, spróbuj: `npx expo start --android`

---

## ETAP 4: Uruchomienie Maestro Studio (Testy E2E)

### Krok 4.1: Upewnij się, że emulator działa z aplikacją
- Emulator musi być uruchomiony (ETAP 2)
- Aplikacja Zdrowie musi być zainstalowana na emulatorze (ETAP 3)

### Krok 4.2: Otwórz Maestro Studio
1. Otwórz menu Start → wpisz **"Maestro Studio"** → kliknij
2. Maestro powinno automatycznie wykryć emulator
3. Jeśli nie — kliknij **"No device connected"** → wybierz emulator z listy

### Krok 4.3: Uruchom testy E2E
Testy E2E znajdują się w folderze:
```
c:\od_zera_do_ai\Zdrowie_v1\zdrowiev1\apps\mobile\__e2e__\
```

Dostępne testy:
| Plik | Co testuje |
|------|-----------|
| `login.yaml` | Logowanie demo użytkownika |
| `login-fail.yaml` | Logowanie z błędnymi danymi |
| `dashboard.yaml` | Widok dashboardu po zalogowaniu |
| `log-meal.yaml` | Dodawanie posiłku |
| `navigation.yaml` | Przechodzenie między zakładkami |

W Maestro Studio:
1. Kliknij **"Open Flow"** → wybierz plik `.yaml` z folderu `__e2e__`
2. Kliknij **"Run"** → obserwuj jak test wykonuje się na emulatorze

---

## ETAP 5: Uruchomienie Testów Jednostkowych (Jest)

Testy jednostkowe nie wymagają emulatora — działają natychmiast:

```powershell
cd c:\od_zera_do_ai\Zdrowie_v1\zdrowiev1\apps\mobile
npm test
```

To uruchomi **53 scenariusze testowych** obejmujące:
- Hooki: `useAuth`, `useHealthData`, `useDietData`, `useBLE`
- Serwisy: `storage`, `api`
- Komponenty: `HealthScoreRing`, `MetricCard`, `SparklineChart`
- Ekrany: `HomeScreen`, `HealthScreen`, `DietScreen`, `OcrScreen`, `ProfileScreen`, `LoginScreen`, `ScaleScreen`

---

## 🚀 Skrócona Ściąga — Codzienny Workflow

```
┌───────────────────────────────────────────────────────┐
│  1. Uruchom emulator Android                          │
│     → Android Studio → Virtual Device → ▶️ Play       │
│                                                       │
│  2. Uruchom API backend (jeśli nie działa)             │
│     cd zdrowiev1                                      │
│     npm run dev (z poziomu turbo)                     │
│                                                       │
│  3. Uruchom Expo na emulatorze                        │
│     cd apps/mobile                                    │
│     npx expo start → naciśnij 'a'                    │
│                                                       │
│  4. (Opcjonalnie) Testy E2E                           │
│     Otwórz Maestro Studio → Open Flow → Run           │
│                                                       │
│  5. (Opcjonalnie) Testy jednostkowe                   │
│     cd apps/mobile && npm test                        │
└───────────────────────────────────────────────────────┘
```

---

## ❌ Najczęstsze Problemy

| Problem | Rozwiązanie |
|---------|-------------|
| `adb devices` nic nie wyświetla | Upewnij się że emulator jest uruchomiony, zamknij i otwórz nowy terminal |
| `ANDROID_HOME is not set` | Uruchom skrypt z Kroku 1.3, zamknij terminal i otwórz nowy |
| Expo: `Could not connect to Android emulator` | Uruchom emulator PRZED `npx expo start` |
| Maestro: `No device connected` | Kliknij przycisk refresh, upewnij się że emulator działa |
| NSIS Error przy Maestro | Pobierz Maestro ponownie — plik się uszkodził przy pobieraniu |
| `npm test` pada na BLE | Upewnij się że `jest.mock('react-native-ble-plx')` jest na górze pliku testowego |

---

## 📁 Struktura Plików Mobile

```
apps/mobile/
├── app.json                      ← Konfiguracja Expo
├── package.json                  ← Zależności (Expo 54, RN 0.81)
├── src/
│   ├── screens/                  ← 7 ekranów aplikacji
│   ├── components/               ← Komponenty UI (karty, wykresy, etc.)
│   ├── hooks/                    ← Logika biznesowa (auth, BLE, health, diet)
│   ├── services/                 ← API client + AsyncStorage
│   ├── navigation/               ← RootNavigator + TabNavigator
│   └── theme/                    ← ⚠️ Do uzupełnienia (design tokens)
└── __e2e__/                      ← Testy Maestro (YAML)
```
