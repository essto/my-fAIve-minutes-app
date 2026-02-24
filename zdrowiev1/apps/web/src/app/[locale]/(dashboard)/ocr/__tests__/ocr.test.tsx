import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import OCRPage from '../page';
import '@testing-library/jest-dom';

describe('OCR Page', () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        window.URL.createObjectURL = vi.fn().mockReturnValue('blob:mock');
    });

    test('renders initial dropzone using a Card component', () => {
        const { container } = render(<OCRPage />);
        expect(screen.getByText('Przeciągnij i upuść dokument')).toBeInTheDocument();

        // The dropzone should be a card
        const card = container.querySelector('[class*="card"]');
        expect(card).toBeInTheDocument();
    });

    test('previews image using Card and Button', async () => {
        const file = new File(['test'], 'test.png', { type: 'image/png' });
        const { container } = render(<OCRPage />);

        const input = screen.getByTestId('dropzone-input');
        await act(async () => {
            await userEvent.upload(input, file);
        });

        // Wait for preview area to appear
        const previewBtn = await screen.findByText('Rozpocznij analizę OCR');
        expect(previewBtn).toBeInTheDocument();

        // Must be a Button component
        const buttons = container.querySelectorAll('button[class*="button_"]');
        expect(buttons.length).toBeGreaterThan(0);

        // Should have a Card
        expect(screen.getByText('Podgląd przesłanego pliku')).toBeInTheDocument();
    });

    test('uploads and displays results within a Card', async () => {
        const mockResult = { original: 'Test', editable: true, values: ['Wynik: Cholesterol 180', 'Glukoza 90'] };
        vi.spyOn(global, 'fetch').mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve(mockResult),
        } as Response);

        const file = new File(['test'], 'test.png', { type: 'image/png' });
        const { container } = render(<OCRPage />);

        const input = screen.getByTestId('dropzone-input');
        await act(async () => {
            await userEvent.upload(input, file);
        });

        // Click start analysis
        const startBtn = await screen.findByText('Rozpocznij analizę OCR');
        await act(async () => {
            await userEvent.click(startBtn);
        });

        // Wait for results to appear
        const extractTitle = await screen.findByText('Wyniki ekstrakcji danych');
        expect(extractTitle).toBeInTheDocument();
        expect(screen.getByText('Wynik: Cholesterol 180')).toBeInTheDocument();

        // Ensure result section uses generic Button component
        expect(screen.getByText('Edytuj dane')).toBeInTheDocument();
        const actionBtn = screen.getByText('Zatwierdź i zapisz w profilu');
        expect(actionBtn).toBeInTheDocument();

        // Must be inside a Card
        const cards = container.querySelectorAll('[class*="card"]');
        expect(cards.length).toBeGreaterThan(1);
    });
});
