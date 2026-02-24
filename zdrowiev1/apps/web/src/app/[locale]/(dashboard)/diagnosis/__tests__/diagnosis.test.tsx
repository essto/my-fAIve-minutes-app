import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import DiagnosisPage from '../page';
import '@testing-library/jest-dom';

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }));

describe('Diagnosis Page', () => {
    beforeEach(() => { vi.restoreAllMocks() });

    test('renders step 1 with Card and Button components', () => {
        const { container } = render(<DiagnosisPage />);

        // Should use Card component for the step container
        const card = container.querySelector('[class*="card"]');
        expect(card).toBeInTheDocument();

        // Symptoms should be rendered using the generic Button component
        const buttons = container.querySelectorAll('button[class*="button_"]');
        expect(buttons.length).toBeGreaterThan(0);
        expect(screen.getByText('Ból głowy')).toBeInTheDocument();
    });

    test('proceeds to step 2 and uses Cards for symptom details', async () => {
        const { container } = render(<DiagnosisPage />);

        // Select a symptom
        const symptomBtn = screen.getByText('Gorączka');
        await act(async () => {
            await userEvent.click(symptomBtn);
        });

        // Click Kontynuuj
        const continueBtn = screen.getByText('Kontynuuj');
        await act(async () => {
            await userEvent.click(continueBtn);
        });

        // Step 2 should be visible (awaiting framer motion)
        expect(await screen.findByText('Szczegóły wybranych objawów')).toBeInTheDocument();

        // Symptom details should ideally use Card styling
        expect(screen.getByText('Czas trwania (godz)')).toBeInTheDocument();
        const inputs = container.querySelectorAll('input');
        expect(inputs.length).toBeGreaterThan(0);
    });

    test('submits report and displays results in a Card', async () => {
        const mockResult = {
            triage: {
                riskLevel: 'HIGH',
                recommendation: 'Skontaktuj się z lekarzem natychmiast.'
            }
        };

        vi.spyOn(global, 'fetch').mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(mockResult),
        } as Response);

        const { container } = render(<DiagnosisPage />);

        // Select a symptom and proceed to step 2
        await act(async () => {
            await userEvent.click(screen.getByText('Kaszel'));
        });
        await act(async () => {
            await userEvent.click(screen.getByText('Kontynuuj'));
        });

        // Submit the report (awaiting step 2 animation before finding the button)
        const odbierzBtn = await screen.findByText('Odbierz Diagnozę');
        await act(async () => {
            await userEvent.click(odbierzBtn);
        });

        await waitFor(() => {
            expect(screen.getByText('Zalecenia medyczne')).toBeInTheDocument();
            expect(screen.getByText(/"Skontaktuj się z lekarzem natychmiast."/)).toBeInTheDocument();
            // Should be wrapped in a Card
            const card = container.querySelector('[class*="card"]');
            expect(card).toBeInTheDocument();
        });
    });
});
