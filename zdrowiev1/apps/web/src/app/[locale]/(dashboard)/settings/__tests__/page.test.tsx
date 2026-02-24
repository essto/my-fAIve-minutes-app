import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import '@testing-library/jest-dom';
import SettingsPage from '../page';

const mockPush = vi.fn();
const mockToggleTheme = vi.fn();
const mockReplace = vi.fn();

vi.mock('next/navigation', () => ({
    useRouter: () => ({
        push: mockPush,
        replace: mockReplace,
    }),
    usePathname: () => '/pl/settings',
}));

vi.mock('next-intl', () => ({
    useTranslations: () => (key: string) => key,
    useLocale: () => 'pl',
}));

vi.mock('next-themes', () => ({
    useTheme: () => ({
        theme: 'dark',
        setTheme: mockToggleTheme,
        themes: ['light', 'dark', 'system'],
    }),
}));

describe('Settings Page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        localStorage.clear();
        global.fetch = vi.fn().mockResolvedValue({
            ok: true,
            json: async () => ({ success: true })
        });
    });

    test('renders all sections', () => {
        render(<SettingsPage />);
        // Sections defined in instructions: Profile, Language, Theme, Notifications, Devices
        expect(screen.getByText('profile_section')).toBeInTheDocument();
        expect(screen.getByText('language_section')).toBeInTheDocument();
        expect(screen.getByText('theme_section')).toBeInTheDocument();
        expect(screen.getByText('notifications_section')).toBeInTheDocument();
        expect(screen.getByText('devices_section')).toBeInTheDocument();
    });

    test('theme toggle switches dark/light', () => {
        render(<SettingsPage />);
        // Clicking light theme
        fireEvent.click(screen.getByRole('button', { name: /light_theme/i }));
        expect(mockToggleTheme).toHaveBeenCalledWith('light');
    });

    test('language preference saved', async () => {
        render(<SettingsPage />);
        // Assuming we have buttons or a select for language
        fireEvent.change(screen.getByRole('combobox', { name: /language/i }), { target: { value: 'en' } });

        await waitFor(() => {
            expect(mockReplace).toHaveBeenCalledWith('/en/settings');
            // Mock router should be called to replace route with new locale.
        });
    });
});
