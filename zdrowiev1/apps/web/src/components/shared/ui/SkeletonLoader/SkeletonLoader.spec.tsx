import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { SkeletonLoader } from './SkeletonLoader';

describe('SkeletonLoader', () => {
    it('renders with default class', () => {
        const { container } = render(<SkeletonLoader />);
        expect(container.firstChild).toHaveClass(/skeleton/);
        // It should have aria-hidden to not confuse screen readers
        expect(container.firstChild).toHaveAttribute('aria-hidden', 'true');
    });

    it('applies custom className', () => {
        const { container } = render(<SkeletonLoader className="h-10 w-full" />);
        expect(container.firstChild).toHaveClass('h-10 w-full');
    });

    it('renders as a circle when variant is circle', () => {
        const { container } = render(<SkeletonLoader variant="circle" />);
        expect(container.firstChild).toHaveClass(/circle/);
    });
});
