import React from 'react';
import styles from './SkeletonLoader.module.css';

export interface SkeletonLoaderProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: 'rectangular' | 'circle';
}

export const SkeletonLoader = React.forwardRef<HTMLDivElement, SkeletonLoaderProps>(
    ({ className, variant = 'rectangular', ...props }, ref) => {
        const baseClass = styles.skeleton;
        const variantClass = variant === 'circle' ? styles.circle : '';
        const shimmerClass = 'shimmer'; // using global animation class from animations.css

        const combinedClassName = `${baseClass} ${variantClass} ${shimmerClass} ${className || ''}`.trim();

        return (
            <div
                ref={ref}
                className={combinedClassName}
                aria-hidden="true"
                {...props}
            />
        );
    }
);
SkeletonLoader.displayName = 'SkeletonLoader';
