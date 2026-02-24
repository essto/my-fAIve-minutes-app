import { render, screen, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import HealthPage from '../page';
import '@testing-library/jest-dom';

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }));

vi.mock('@/components/shared/charts/HealthChart', () => ({
    HealthChart: () => <div data-testid="health-chart-mock">Health Chart</div>
}));

describe('Health Page', () => {
    beforeEach(() => { vi.restoreAllMocks() });

    test('shows SkeletonLoaders on loading instead of animate-pulse divs', () => {
        vi.spyOn(global, 'fetch').mockImplementationOnce(() => new Promise(() => { }));
        const { container } = render(<HealthPage />);
        const skeletons = container.querySelectorAll('[class*="skeleton"]');
        expect(skeletons.length).toBeGreaterThan(0);
        // Ensure no old animate-pulse classes
        expect(container.querySelectorAll('.animate-pulse').length).toBe(0);
    });

    test('displays Health page with Cards and generic action Button after loading', async () => {
        const mockData = {
            metrics: {
                heartRate: { current: 72, avg7d: 70, min: 60, max: 120 },
                sleep: { lastNight: '7h 30m', avg7d: '7h 15m', quality: 'Dobra' },
                weight: { current: 75.5, change30d: -0.5, bmi: 24.1 }
            },
            charts: {
                heartRateHistory: { type: 'line', data: [], keys: [], colors: [] },
                sleepHistory: { type: 'bar', data: [], keys: [], colors: [] },
                weightHistory: { type: 'line', data: [], keys: [], colors: [] }
            }
        };

        vi.spyOn(global, 'fetch').mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(mockData),
        } as Response);

        const { container } = render(<HealthPage />);

        await waitFor(() => {
            // Should contain cards for metrics and charts
            const cards = container.querySelectorAll('[class*="card"]');
            expect(cards.length).toBeGreaterThan(0);

            // Should contain mocked metrics
            expect(screen.getByText('72')).toBeInTheDocument();
            expect(screen.getByText('7h 30m')).toBeInTheDocument();

            // Should contain a generic action Button using the new unified Button component
            const buttons = container.querySelectorAll('button[class*="button_"]');
            expect(buttons.length).toBeGreaterThan(0);
            expect(screen.getByText(/Udostępnij raport|Pobierz raport/i)).toBeInTheDocument();
        });
    });
});
