export const designTokens = {
    colors: {
        primary: '#0F172A',
        secondary: '#10B981',
        background: '#FFFFFF',
        text: '#1F2937',
        error: '#EF4444',
    },
    typography: {
        fontFamily: 'Inter, sans-serif',
        fontSize: {
            sm: '0.875rem',
            base: '1rem',
            lg: '1.125rem',
            xl: '1.25rem',
        },
    },
    spacing: {
        1: '0.25rem',
        2: '0.5rem',
        4: '1rem',
        8: '2rem',
    },
} as const;

export type DesignTokens = typeof designTokens;
