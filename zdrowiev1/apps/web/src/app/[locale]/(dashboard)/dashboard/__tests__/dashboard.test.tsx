import { render, screen, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import Dashboard from '../page';
import '@testing-library/jest-dom';

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }));

vi.mock('@/components/shared/charts/HealthChart', () => ({
    HealthChart: () => <div data-testid="health-chart-mock">Health Chart</div>
}));

describe('Dashboard Page', () => {
    beforeEach(() => { vi.restoreAllMocks() });

    test('shows premium SkeletonLoaders on loading', () => {
        vi.spyOn(global, 'fetch').mockImplementationOnce(() => new Promise(() => { }));
        const { container } = render(<Dashboard />);
        // Ensure new SkeletonLoader class is used, checking partial class name due to CSS modules
        const skeletons = container.querySelectorAll('[class*="skeleton"]');
        expect(skeletons.length).toBeGreaterThan(0);
    });

    test('displays dashboard with Cards and Health Score ring after loading', async () => {
        const mockData = {
            healthScore: 85,
            anomalies: [
                { metric: 'Wysoki poziom cukru', value: 120, severity: 'high', message: 'Zalecana konsultacja' },
                { metric: 'Brak aktywności', value: 2000, severity: 'medium', message: 'Zrób 8 tys. kroków' }
            ],
            charts: {
                healthTrend: { type: 'line', data: [], keys: [], colors: [] },
                activityRings: { type: 'progress_ring', data: [], keys: [], colors: [] },
                sleepQuality: { type: 'bar', data: [], keys: [], colors: [] }
            }
        };

        vi.spyOn(global, 'fetch').mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(mockData),
        } as Response);

        const { container } = render(<Dashboard />);

        await waitFor(() => {
            // Should contain cards
            const cards = container.querySelectorAll('[class*="card"]');
            expect(cards.length).toBeGreaterThan(0);

            // Should contain the Health Score (in the premium design format)
            expect(screen.getByText(/85/)).toBeInTheDocument();
            expect(screen.getByText(/Twój wynik zdrowia/i)).toBeInTheDocument();

            // Anomalies are displayed
            expect(screen.getByText(/Wysoki poziom cukru/)).toBeInTheDocument();
        });
    });
});
