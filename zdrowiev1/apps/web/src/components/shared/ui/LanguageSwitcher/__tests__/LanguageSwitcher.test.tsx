import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import '@testing-library/jest-dom';
import { LanguageSwitcher } from '../LanguageSwitcher';

const mockReplace = vi.fn();

// Mock useRouter
vi.mock('next/navigation', () => ({
    useRouter: () => ({
        replace: mockReplace,
        push: vi.fn(),
    }),
    usePathname: () => '/dashboard',
    useParams: () => ({}),
}));

vi.mock('next-intl', () => ({
    useLocale: () => 'pl',
    useTranslations: () => (key: string) => key,
}));

describe('LanguageSwitcher', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    test('renders current language indicator', () => {
        render(<LanguageSwitcher />);
        expect(screen.getByText('PL')).toBeInTheDocument();
    });

    test('switching language calls router.replace', async () => {
        render(<LanguageSwitcher />);
        fireEvent.click(screen.getByRole('button'));

        await waitFor(() => {
            expect(mockReplace).toHaveBeenCalledWith('/en/dashboard');
        });
    });
});
