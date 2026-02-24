import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { Button } from './Button';

describe('Button', () => {
    it('renders children correctly', () => {
        render(<Button>Click me</Button>);
        expect(screen.getByRole('button', { name: /click me/i })).toBeInTheDocument();
    });

    it('handles click events', async () => {
        const handleClick = vi.fn();
        render(<Button onClick={handleClick}>Click me</Button>);

        const button = screen.getByRole('button', { name: /click me/i });
        await userEvent.click(button);

        expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('can be disabled', async () => {
        const handleClick = vi.fn();
        render(<Button disabled onClick={handleClick}>Click me</Button>);

        const button = screen.getByRole('button', { name: /click me/i });
        expect(button).toBeDisabled();

        await userEvent.click(button);
        expect(handleClick).not.toHaveBeenCalled();
    });

    it('applies variant classes correctly', () => {
        render(<Button variant="secondary">Secondary</Button>);
        // We will check if it has the secondary class from CSS modules
        // Using a data attribute or class name matching
        const button = screen.getByRole('button');
        expect(button.className).toContain('secondary');
    });

    it('shows loading state mechanism', () => {
        render(<Button isLoading>Submit</Button>);
        const button = screen.getByRole('button');
        expect(button).toBeDisabled();
        expect(button).toHaveAttribute('data-loading', 'true');
        // Shouldn't show "Submit" text directly if we use a spinner, or maybe alongside it
        expect(screen.getByText('Submit')).toBeInTheDocument();
        expect(button.className).toContain('loading');
    });
});
