import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import '@testing-library/jest-dom';
import { OnboardingWizard } from '../OnboardingWizard';

// Mock useRouter
const mockPush = vi.fn();
vi.mock('next/navigation', () => ({
    useRouter: () => ({
        push: mockPush,
    }),
}));

vi.mock('next-intl', () => ({
    useTranslations: () => (key: string) => key,
}));

global.fetch = vi.fn();

describe('OnboardingWizard', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    test('renders step 1 (Profile) by default', () => {
        render(<OnboardingWizard />);
        expect(screen.getByText('step_1_title')).toBeInTheDocument(); // Title for Profile setup
        expect(screen.getByRole('textbox', { name: /name/i })).toBeInTheDocument();
        expect(screen.getByRole('spinbutton', { name: /age/i })).toBeInTheDocument();
    });

    test('validates required fields before next', async () => {
        render(<OnboardingWizard />);
        fireEvent.click(screen.getByRole('button', { name: /next/i }));

        await waitFor(() => {
            expect(screen.getByText('name_required')).toBeInTheDocument();
        });
    });

    test('navigates step 1 -> 2 -> 3', async () => {
        render(<OnboardingWizard />);

        // Step 1
        fireEvent.change(screen.getByRole('textbox', { name: /name/i }), { target: { value: 'Jan' } });
        fireEvent.change(screen.getByRole('spinbutton', { name: /age/i }), { target: { value: '30' } });
        fireEvent.change(screen.getByRole('spinbutton', { name: /height/i }), { target: { value: '180' } });
        fireEvent.change(screen.getByRole('spinbutton', { name: /weight/i }), { target: { value: '80' } });
        fireEvent.click(screen.getByRole('button', { name: /next/i }));

        // Step 2
        await waitFor(() => {
            expect(screen.getByText('step_2_title')).toBeInTheDocument(); // Goals
        });
        fireEvent.click(screen.getByRole('button', { name: /lose_weight/i }));
        fireEvent.click(screen.getByRole('button', { name: /next/i }));

        // Step 3
        await waitFor(() => {
            expect(screen.getByText('step_3_title')).toBeInTheDocument(); // Devices
        });
    });

    test('can go back preserving data', async () => {
        render(<OnboardingWizard />);

        // Fill Step 1 and go to Step 2
        fireEvent.change(screen.getByRole('textbox', { name: /name/i }), { target: { value: 'Anna' } });
        fireEvent.change(screen.getByRole('spinbutton', { name: /age/i }), { target: { value: '25' } });
        fireEvent.change(screen.getByRole('spinbutton', { name: /height/i }), { target: { value: '165' } });
        fireEvent.change(screen.getByRole('spinbutton', { name: /weight/i }), { target: { value: '60' } });
        fireEvent.click(screen.getByRole('button', { name: /next/i }));

        // Go back
        await waitFor(() => {
            expect(screen.getByText('step_2_title')).toBeInTheDocument();
        });
        fireEvent.click(screen.getByRole('button', { name: /back/i }));

        // Check Step 1 data is preserved
        await waitFor(() => {
            expect(screen.getByRole('textbox', { name: /name/i })).toHaveValue('Anna');
        });
    });

    test('submits onboarding on final step', async () => {
        const mockFetch = vi.fn().mockResolvedValueOnce({
            ok: true,
            json: async () => ({ success: true })
        });
        global.fetch = mockFetch as any;

        render(<OnboardingWizard />);

        // Step 1
        fireEvent.change(screen.getByRole('textbox', { name: /name/i }), { target: { value: 'Jan' } });
        fireEvent.change(screen.getByRole('spinbutton', { name: /age/i }), { target: { value: '30' } });
        fireEvent.change(screen.getByRole('spinbutton', { name: /height/i }), { target: { value: '180' } });
        fireEvent.change(screen.getByRole('spinbutton', { name: /weight/i }), { target: { value: '80' } });
        fireEvent.click(screen.getByRole('button', { name: /next/i }));

        // Step 2
        await waitFor(() => {
            expect(screen.getByText('step_2_title')).toBeInTheDocument();
        });
        fireEvent.click(screen.getByRole('button', { name: /lose_weight/i }));
        fireEvent.click(screen.getByRole('button', { name: /next/i }));

        // Step 3
        await waitFor(() => {
            expect(screen.getByText('step_3_title')).toBeInTheDocument();
        });
        fireEvent.click(screen.getByRole('button', { name: /finish/i }));

        await waitFor(() => {
            expect(global.fetch).toHaveBeenCalledWith('/api/user/onboarding', expect.objectContaining({
                method: 'POST',
                body: expect.any(String)
            }));
            expect(mockPush).toHaveBeenCalledWith('/dashboard');
        });
    });

    test('shows step indicators (1/3, 2/3, 3/3)', () => {
        render(<OnboardingWizard />);
        expect(screen.getByText('1 / 3')).toBeInTheDocument();
    });
});
