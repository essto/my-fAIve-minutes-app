/* apps/web/src/components/shared/layout/AppShell.tsx */
import React from 'react';
import { Sidebar } from './Sidebar';
import styles from './AppShell.module.css';

interface AppShellProps {
    children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
    return (
        <div className={styles.shell}>
            <Sidebar />
            <main className={styles.main}>
                {children}
            </main>
        </div>
    );
}
