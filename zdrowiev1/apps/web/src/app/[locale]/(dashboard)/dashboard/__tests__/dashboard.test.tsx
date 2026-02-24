import { render, screen, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import Dashboard from '../page';
import '@testing-library/jest-dom';

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }), usePathname: () => '/' }));

vi.mock('next-intl', () => ({
    useTranslations: () => (key: string) => key,
    useLocale: () => 'pl'
}));

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

        // Setup localStorage BEFORE render so onboarding check passes
        const localStorageMock = { getItem: vi.fn((key: string) => key === 'onboarding_completed' ? 'true' : null), setItem: vi.fn(), removeItem: vi.fn(), clear: vi.fn(), length: 0, key: vi.fn() };
        Object.defineProperty(window, 'localStorage', { value: localStorageMock, writable: true, configurable: true });

        vi.spyOn(global, 'fetch').mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(mockData),
        } as Response);

        const { container } = render(<Dashboard />);

        // Wait for anomaly to appear (proves data loaded successfully)
        const anomalyText = await screen.findByText(/Wysoki poziom cukru/, {}, { timeout: 5000 });
        expect(anomalyText).toBeInTheDocument();

        // Should contain cards (CSS modules hash the class name)
        const cards = container.querySelectorAll('[class*="card"]');
        expect(cards.length).toBeGreaterThan(0);

        // Health score should be visible (may match health_score and health_score_desc)
        expect(screen.getAllByText(/health_score/i).length).toBeGreaterThan(0);
    });
});
