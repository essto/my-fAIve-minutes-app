import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Sidebar } from './Sidebar';

// Mock next/navigation
vi.mock('next/navigation', () => ({
    usePathname: () => '/dashboard',
}));

// Mock next/link (render as standard anchor tag)
vi.mock('next/link', () => ({
    default: ({ children, href, className }: { children: React.ReactNode, href: string, className?: string }) => (
        <a href={href} className={className}>{children}</a>
    ),
}));

// Mock ThemeToggle component
vi.mock('../ui/ThemeToggle/ThemeToggle', () => ({
    ThemeToggle: () => <button data-testid="theme-toggle">Theme Toggle</button>,
}));

describe('Sidebar', () => {
    it('renders logo and navigation links', () => {
        render(<Sidebar />);
        expect(screen.getByText('Zdrowie App')).toBeInTheDocument();
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Wyloguj')).toBeInTheDocument();
    });

    it('highlights the active link correctly using gradient active state class', () => {
        render(<Sidebar />);
        const activeLink = screen.getByText('Dashboard').closest('a');
        expect(activeLink?.className).toContain('bg-brand');
    });

    it('does not use emoji icons anymore, but renders SVG icons', () => {
        const { container } = render(<Sidebar />);
        // Checking if emojis like 📊 or ❤️ are NOT in the document text
        expect(container.textContent).not.toMatch(/[📊❤️🍎🧠📋]/);

        // Ensure there are SVG elements rendered for the icons
        const svgs = container.querySelectorAll('svg');
        expect(svgs.length).toBeGreaterThan(0);
    });
});
