import React from 'react';
import styles from './Button.module.css';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
    isLoading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ children, variant = 'primary', isLoading = false, className, disabled, ...props }, ref) => {
        const baseClass = styles.button;
        const variantClass = styles[variant] || '';
        const loadingClass = isLoading ? styles.loading : '';
        const combinedClassName = `${baseClass} ${variantClass} ${loadingClass} ${className || ''}`.trim();

        return (
            <button
                ref={ref}
                className={combinedClassName}
                disabled={disabled || isLoading}
                data-loading={isLoading}
                {...props}
            >
                {isLoading && <span className={styles.spinner} aria-hidden="true" />}
                <span className={styles.content}>{children}</span>
            </button>
        );
    }
);

Button.displayName = 'Button';
