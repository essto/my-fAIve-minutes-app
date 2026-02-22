/* apps/web/src/components/shared/ui/ThemeToggle/ThemeToggle.tsx */
'use client';

import React from 'react';
import { useTheme } from '../../../../providers/ThemeProvider';
import styles from './ThemeToggle.module.css';

export function ThemeToggle() {
    const { theme, toggleTheme } = useTheme();

    return (
        <button
            onClick={toggleTheme}
            className={styles.toggle}
            title={theme === 'light' ? 'Przełącz na ciemny motyw' : 'Przełącz na jasny motyw'}
        >
            {theme === 'light' ? '🌙' : '☀️'}
        </button>
    );
}
