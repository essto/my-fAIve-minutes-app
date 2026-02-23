import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { Card, CardHeader, CardTitle, CardContent } from './Card';

describe('Card', () => {
    it('renders basic card with children', () => {
        render(
            <Card>
                <div>Card Body</div>
            </Card>
        );
        expect(screen.getByText('Card Body')).toBeInTheDocument();
        const card = screen.getByText('Card Body').parentElement;
        expect(card?.className).toContain('card'); // internal module class
    });

    it('renders CardTitle and CardHeader properly', () => {
        render(
            <Card>
                <CardHeader>
                    <CardTitle>My Title</CardTitle>
                </CardHeader>
                <CardContent>
                    <p>Some content</p>
                </CardContent>
            </Card>
        );
        expect(screen.getByText('My Title')).toBeInTheDocument();
        expect(screen.getByText('Some content')).toBeInTheDocument();
        // Title should be h3 or similar heading
        expect(screen.getByRole('heading', { name: /my title/i, level: 3 })).toBeInTheDocument();
    });

    it('applies interactive class if onClick is provided or interactive prop is true', () => {
        const { container, rerender } = render(<Card interactive>Hover me</Card>);
        expect(container.firstChild).toHaveClass(/interactive/);

        rerender(<Card onClick={() => { }}>Click me</Card>);
        expect(container.firstChild).toHaveClass(/interactive/);
    });

    it('applies gradient top border if gradientAccent is true', () => {
        const { container } = render(<Card gradientAccent>Health Stats</Card>);
        expect(container.firstChild).toHaveClass(/gradientAccent/);
    });

    it('applies glassmorphism if glass is true', () => {
        const { container } = render(<Card glass>Transparent</Card>);
        expect(container.firstChild).toHaveClass(/glass/);
    });
});
