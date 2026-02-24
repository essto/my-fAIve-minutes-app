import { render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import '@testing-library/jest-dom';
import SentryErrorBoundary from '../SentryErrorBoundary';
import * as Sentry from '@sentry/nextjs';

vi.mock('@sentry/nextjs', () => ({
    captureException: vi.fn(),
    init: vi.fn(),
}));

const ThrowingComponent = () => {
    throw new Error('Test Error');
};

describe('SentryErrorBoundary', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        // Prevent console.error from scattering in test output
        vi.spyOn(console, 'error').mockImplementation(() => { });
    });

    test('catches error and shows fallback UI', () => {
        render(
            <SentryErrorBoundary>
                <ThrowingComponent />
            </SentryErrorBoundary>
        );
        expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });

    test('reports error to Sentry', () => {
        render(
            <SentryErrorBoundary>
                <ThrowingComponent />
            </SentryErrorBoundary>
        );
        expect(Sentry.captureException).toHaveBeenCalledWith(expect.any(Error));
    });

    test('renders children normally when no error', () => {
        render(
            <SentryErrorBoundary>
                <div>Everything is fine</div>
            </SentryErrorBoundary>
        );
        expect(screen.getByText('Everything is fine')).toBeInTheDocument();
    });
});
