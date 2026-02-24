/* apps/web/src/components/shared/ui/ThemeToggle/ThemeToggle.tsx */
'use client';

import React from 'react';
import { useTheme } from '../../../../providers/ThemeProvider';
import { Moon, Sun } from 'lucide-react';

export function ThemeToggle() {
    const { theme, toggleTheme } = useTheme();

    return (
        <button
            onClick={toggleTheme}
            className="flex items-center justify-center p-2 rounded-lg bg-neutral-bg2 hover:bg-neutral-bg3 border border-border text-foreground transition-all duration-200"
            title={theme === 'light' ? 'Przełącz na ciemny motyw' : 'Przełącz na jasny motyw'}
        >
            {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
        </button>
    );
}
