import { render, screen, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import Dashboard from '../page'
import '@testing-library/jest-dom'

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }))

describe('Dashboard Page', () => {
    beforeEach(() => { vi.restoreAllMocks() })

    test('shows skeleton on loading', () => {
        vi.spyOn(global, 'fetch').mockImplementationOnce(() => new Promise(() => { }))
        render(<Dashboard />)
        expect(screen.getByTestId('dashboard-skeleton')).toBeInTheDocument()
    })

    test('displays data after loading', async () => {
        const mockData = {
            healthScore: 85,
            anomalies: ['Wysoki poziom cukru', 'Brak aktywności'],
            charts: {
                weight: [{ date: '2023-01', value: 80 }, { date: '2023-02', value: 78 }],
                activityRings: { move: 75, sleep: 90, diet: 60 }
            }
        }
        vi.spyOn(global, 'fetch').mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve(mockData),
        } as Response)

        render(<Dashboard />)
        await waitFor(() => {
            expect(screen.getByText('Health Score: 85')).toBeInTheDocument()
            expect(screen.getByText('Wysoki poziom cukru')).toBeInTheDocument()
            expect(screen.getByText('Brak aktywności')).toBeInTheDocument()
            expect(screen.getByTestId('weight-chart')).toBeInTheDocument()
            expect(screen.getByTestId('activity-rings')).toBeInTheDocument()
        })
    })
})
