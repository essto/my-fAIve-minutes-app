import React from 'react';
import styles from './Card.module.css';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
    interactive?: boolean;
    gradientAccent?: boolean;
    glass?: boolean;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
    ({ children, className, interactive, gradientAccent, glass, onClick, ...props }, ref) => {
        const baseClass = styles.card;
        const interactiveClass = interactive || onClick ? styles.interactive : '';
        const accentClass = gradientAccent ? styles.gradientAccent : '';
        const glassClass = glass ? styles.glass : '';

        const combinedClassName = `${baseClass} ${interactiveClass} ${accentClass} ${glassClass} ${className || ''}`.trim();

        return (
            <div ref={ref} className={combinedClassName} onClick={onClick} {...props}>
                {children}
            </div>
        );
    }
);
Card.displayName = 'Card';

export const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
    ({ className, ...props }, ref) => (
        <div ref={ref} className={`${styles.header} ${className || ''}`.trim()} {...props} />
    )
);
CardHeader.displayName = 'CardHeader';

export const CardTitle = React.forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(
    ({ className, ...props }, ref) => (
        <h3 ref={ref} className={`${styles.title} ${className || ''}`.trim()} {...props} />
    )
);
CardTitle.displayName = 'CardTitle';

export const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
    ({ className, ...props }, ref) => (
        <div ref={ref} className={`${styles.content} ${className || ''}`.trim()} {...props} />
    )
);
CardContent.displayName = 'CardContent';
