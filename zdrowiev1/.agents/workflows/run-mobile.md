---
description: How to start the mobile app on Android Emulator for testing
---

## Prerequisites
- Android Studio installed at `C:\Program Files\Android\Android Studio`
- Maestro Studio installed at `C:\Users\pszymelyniec\AppData\Local\Programs\Maestro Studio`
- Android SDK downloaded via Android Studio Setup Wizard
- At least one AVD (Android Virtual Device) created (e.g. Pixel 8 API 34)

## Steps

1. **Open Android Studio and start the emulator**  
   Open Android Studio → More Actions → Virtual Device Manager → click ▶️ Play on your device

2. **Verify the emulator is running**
```powershell
adb devices
```
Expected: `emulator-5554   device`

// turbo
3. **Install mobile dependencies (if needed)**
```powershell
cd c:\od_zera_do_ai\Zdrowie_v1\zdrowiev1\apps\mobile
npx expo install
```

4. **Start the Expo dev server and open on Android**
```powershell
cd c:\od_zera_do_ai\Zdrowie_v1\zdrowiev1\apps\mobile
npx expo start --android
```

5. **Wait for the app to build and appear on the emulator** (first time ~2-3 min)

6. **(Optional) Run unit tests**
```powershell
cd c:\od_zera_do_ai\Zdrowie_v1\zdrowiev1\apps\mobile
npm test
```

7. **(Optional) Run E2E tests with Maestro**  
   Open Maestro Studio → Open Flow → select a `.yaml` file from `apps/mobile/__e2e__/` → Run
