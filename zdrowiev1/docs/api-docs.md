# API Documentation

Poniższy dokument opisuje najważniejsze zbiory endpointów w ramach architektury `Zdrowie App`. System oparty jest na NestJS z włączonym autoryzacyjnym zabezpieczeniem na poziomie kontrolerów.

## Ogólna Struktura i Format

Wszystkie requesty do API i odpowiedzi używają formatu `application/json`.
Gdy wystąpi błąd, odpowiedź posiada ogólny format:

```json
{
  "statusCode": 400,
  "message": "Opis błędu",
  "error": "Bad Request"
}
```

## Authentication

### `POST /api/auth/login`

Autoryzacja użytkownika, zwraca token JWT.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1...",
  "user": {
    "id": "123",
    "email": "user@example.com",
    "name": "Jan Kowalski"
  }
}
```

## Dashboard & Metrics

### `GET /api/dashboard`

Pobiera dane dla głównego pulpitu wraz ze zsumowanym Health Score oraz wytycznymi dotyczącymi anomalii.

**Headers:**
`Authorization: Bearer <token>`

**Response (200 OK):**
```json
{
  "healthScore": 85,
  "anomalies": [
    {
      "metric": "sleep",
      "status": "warning",
      "message": "Niska jakość snu przez 2 dni"
    }
  ],
  "trends": {
    "heartRate": "+2%",
    "weight": "-1.5kg"
  }
}
```

## Onboarding

### `POST /api/user/onboarding`

Zapisuje dane początkowe zdefiniowane w Onboarding Wizardzie.

**Request:**
```json
{
  "profile": {
    "name": "Jan",
    "age": 30,
    "height": 180,
    "weight": 80
  },
  "goals": {
    "goal": "lose_weight"
  },
  "devices": {
    "devices": ["apple_health", "garmin"]
  }
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "onboardingCompleted": true
}
```

## Integracje Urządzeń (BLE / Health APIs)

### `GET /api/devices/status`

Status powiązanych zewnętrznych źródeł danych.

*Więcej endpointów m.in. dla OCR i Diagnostyki AI zostanie wygenerowanych poprzez konfigurację OpenAPI/Swagger dostępną standardowo poprzez `/api/docs`.*
