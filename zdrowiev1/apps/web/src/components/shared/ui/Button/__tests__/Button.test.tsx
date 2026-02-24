import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Button } from '../Button';

describe('Button Component', () => {
    it('should render children correctly', () => {
        render(<Button>Click me</Button>);
        expect(screen.getByText('Click me')).toBeDefined();
    });

    it('should trigger onClick handler when clicked', () => {
        const handleClick = vi.fn();
        render(<Button onClick={handleClick}>Click me</Button>);

        fireEvent.click(screen.getByRole('button', { name: /click me/i }));
        expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('should be disabled when isLoading is true', () => {
        render(<Button isLoading>Loading</Button>);
        const button = screen.getByRole('button');
        expect(button).toHaveProperty('disabled', true);

        // Test data attribute (if supported by environment) or simply finding the disabled status
        expect(button.getAttribute('data-loading')).toBe('true');
    });

    it('should not be clickable if disabled is true', () => {
        const handleClick = vi.fn();
        render(<Button onClick={handleClick} disabled>Disabled</Button>);

        const button = screen.getByRole('button');
        fireEvent.click(button);
        expect(handleClick).not.toHaveBeenCalled();
        expect(button).toHaveProperty('disabled', true);
    });
});
