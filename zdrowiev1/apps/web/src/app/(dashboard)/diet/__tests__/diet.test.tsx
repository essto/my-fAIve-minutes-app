import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import DietPage from '../page';
import '@testing-library/jest-dom';

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }));

describe('Diet Page', () => {
    beforeEach(() => { vi.restoreAllMocks() });

    test('shows SkeletonLoaders on loading instead of generic divs', () => {
        vi.spyOn(global, 'fetch').mockImplementationOnce(() => new Promise(() => { }));
        const { container } = render(<DietPage />);
        // Checking for our SkeletonLoader instances
        const skeletons = container.querySelectorAll('[class*="skeleton"]');
        expect(skeletons.length).toBeGreaterThan(0);
        // Ensure old tailwind animate-pulse classes are removed
        expect(container.querySelectorAll('.animate-pulse').length).toBe(0);
    });

    test('displays Diet details using Cards and unified Buttons', async () => {
        const mockSummary = {
            total: { calories: 1500, protein: 90, carbs: 120, fat: 40 },
            target: { calories: 2000, protein: 120, carbs: 200, fat: 60 },
            isDeficit: true,
            isSurplus: false
        };

        vi.spyOn(global, 'fetch').mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(mockSummary),
        } as Response);

        const { container } = render(<DietPage />);

        await waitFor(() => {
            // Wait for summary data to load
            expect(screen.getByText('1500')).toBeInTheDocument();
            expect(screen.getByText(/2000 kcal/)).toBeInTheDocument();

            // Should contain cards for macros
            const cards = container.querySelectorAll('[class*="card"]');
            expect(cards.length).toBeGreaterThan(0);

            // The main action 'Zaloguj Posiłek' should be using the unified Button component
            const buttons = container.querySelectorAll('button[class*="button_"]');
            expect(buttons.length).toBeGreaterThan(0);
        });
    });

    test('opens modal using Card when Zaloguj Posiłek is clicked', async () => {
        const mockSummary = {
            total: { calories: 1500, protein: 90, carbs: 120, fat: 40 },
            target: { calories: 2000, protein: 120, carbs: 200, fat: 60 },
            isDeficit: true,
            isSurplus: false
        };

        vi.spyOn(global, 'fetch').mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(mockSummary),
        } as Response);

        const { container } = render(<DietPage />);

        await waitFor(() => {
            expect(screen.getByText('Zaloguj Posiłek')).toBeInTheDocument();
        });

        // Click the button to open modal
        const logMealButton = screen.getByRole('button', { name: /Zaloguj Posiłek/i });
        await act(async () => {
            await userEvent.click(logMealButton);
        });

        // Modal should appear
        const modalTitle = screen.getByText('Zaloguj nowy posiłek');
        expect(modalTitle).toBeInTheDocument();

        // Modal should preferably use the Card component for consistency
        const modalContainer = modalTitle.closest('[class*="card"]');
        expect(modalContainer).toBeInTheDocument();
    });
});
