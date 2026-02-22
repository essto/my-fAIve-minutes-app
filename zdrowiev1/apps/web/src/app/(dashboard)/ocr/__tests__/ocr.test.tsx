import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import OCRPage from '../page'
import '@testing-library/jest-dom'

describe('OCR Page', () => {
    beforeEach(() => {
        vi.restoreAllMocks()
        window.URL.createObjectURL = vi.fn().mockReturnValue('blob:mock')
    })

    test('renders dropzone', () => {
        render(<OCRPage />)
        expect(screen.getByText('Przeciągnij plik tutaj')).toBeInTheDocument()
        expect(screen.getByText('lub kliknij aby wybrać')).toBeInTheDocument()
    })

    test('previews image after selection', async () => {
        const file = new File(['test'], 'test.png', { type: 'image/png' })
        render(<OCRPage />)
        const input = screen.getByTestId('dropzone-input')
        fireEvent.change(input, { target: { files: [file] } })
        await waitFor(() => {
            expect(screen.getByTestId('file-preview')).toBeInTheDocument()
        })
    })

    test('uploads and displays results', async () => {
        const mockResult = { original: 'Test', editable: true, values: ['Result1', 'Result2'] }
        vi.spyOn(global, 'fetch').mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve(mockResult),
        } as Response)

        const file = new File(['test'], 'test.png', { type: 'image/png' })
        render(<OCRPage />)
        const input = screen.getByTestId('dropzone-input')
        fireEvent.change(input, { target: { files: [file] } })
        await waitFor(() => screen.getByText('Prześlij do analizy'))
        fireEvent.click(screen.getByText('Prześlij do analizy'))

        await waitFor(() => {
            expect(global.fetch).toHaveBeenCalled()
            expect(screen.getByText('Result1')).toBeInTheDocument()
            expect(screen.getByText('Edytuj wartości')).toBeInTheDocument()
        })
    })
})
