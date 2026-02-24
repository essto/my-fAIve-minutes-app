import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { AppShell } from './AppShell';

// Mock dependencies
vi.mock('./Sidebar', () => ({
    Sidebar: () => <div data-testid="sidebar">Sidebar Mock</div>
}));
vi.mock('next/link', () => ({
    default: ({ children, href, className }: { children: React.ReactNode, href: string, className?: string }) => <a href={href} className={className}>{children}</a>
}));
vi.mock('next/navigation', () => ({
    usePathname: () => '/dashboard'
}));
vi.mock('../ui/ThemeToggle/ThemeToggle', () => ({
    ThemeToggle: () => <button data-testid="theme-toggle">Theme Toggle Mock</button>
}));

describe('AppShell', () => {
    it('renders children and the sidebar', () => {
        render(<AppShell><div>Test Content</div></AppShell>);
        expect(screen.getByText('Test Content')).toBeInTheDocument();
        expect(screen.getByTestId('sidebar')).toBeInTheDocument();
    });

    it('renders a mobile top header', () => {
        render(<AppShell><div>Content</div></AppShell>);
        // Expect a header element that likely contains the mobile logo and maybe a profile/menu button
        const header = screen.getByRole('banner');
        expect(header).toBeInTheDocument();
        // Check if there is some title text or logo inside header
        expect(header).toHaveTextContent(/Zdrowie/);
    });

    it('renders mobile bottom navigation', () => {
        render(<AppShell><div>Content</div></AppShell>);
        // Expect a nav element explicitly for mobile (can be identified by class or standard nav)
        // Since Sidebar might have a nav, there should be at least two navs or a specific mobile block 
        // We'll look for bottom nav specific links (like 'Strona główna' or matching hrefs)
        const mobileNavContainer = screen.getByTestId('mobile-bottom-nav');
        expect(mobileNavContainer).toBeInTheDocument();
        // Check if basic links exist
        expect(mobileNavContainer.querySelector('a[href="/dashboard"]')).toBeInTheDocument();
    });
});
