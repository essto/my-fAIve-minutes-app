import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import DiagnosisPage from '../page'
import '@testing-library/jest-dom'

describe('Diagnosis Page', () => {
    beforeEach(() => { vi.restoreAllMocks() })

    test('renders symptoms list', () => {
        render(<DiagnosisPage />)
        expect(screen.getByText('Ból głowy')).toBeInTheDocument()
        expect(screen.getByText('Gorączka')).toBeInTheDocument()
        expect(screen.getByText('Zawroty głowy')).toBeInTheDocument()
        expect(screen.getByText('Sprawdź')).toBeInTheDocument()
    })

    test('shows results after diagnosis', async () => {
        const mockResult = { triage: 'ŻÓŁTY', description: 'Konsultacja lekarska w ciągu 24h' }
        vi.spyOn(global, 'fetch').mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve(mockResult),
        } as Response)

        render(<DiagnosisPage />)
        fireEvent.click(screen.getByLabelText('Ból głowy'))
        fireEvent.click(screen.getByText('Sprawdź'))

        await waitFor(() => {
            expect(global.fetch).toHaveBeenCalledWith('/api/diagnosis/check', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symptoms: ['Ból głowy'] })
            })
            expect(screen.getByText('Konsultacja lekarska w ciągu 24h')).toBeInTheDocument()
            expect(screen.getByText('ŻÓŁTY')).toHaveClass('bg-yellow-500')
        })
    })
})
