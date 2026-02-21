import { http, HttpResponse } from 'msw'

export const handlers = [
    http.post('/api/auth/login', () =>
        HttpResponse.json({ token: 'mock-token' })
    ),
    http.get('/api/visualization/dashboard', () =>
        HttpResponse.json({
            healthScore: 85,
            anomalies: ['Wysoki poziom cukru', 'Brak aktywności'],
            charts: {
                weight: [{ date: '2023-01', value: 80 }, { date: '2023-02', value: 78 }],
                activityRings: { move: 75, sleep: 90, diet: 60 }
            }
        })
    ),
    http.post('/api/ocr/upload', () =>
        HttpResponse.json({ original: 'Przykładowy wynik OCR', editable: true, values: ['Wynik 1: 120/80', 'Wynik 2: 70 kg'] })
    ),
    http.post('/api/diagnosis/check', () =>
        HttpResponse.json({ triage: 'ŻÓŁTY', description: 'Konsultacja lekarska w ciągu 24h' })
    ),
]
