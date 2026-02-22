/* apps/web/src/components/shared/layout/Sidebar.tsx */
'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import styles from './Sidebar.module.css';
import { ThemeToggle } from '../ui/ThemeToggle/ThemeToggle';

const navItems = [
    { name: 'Dashboard', href: '/dashboard', icon: '📊' },
    { name: 'Zdrowie', href: '/health', icon: '❤️' },
    { name: 'Dieta', href: '/diet', icon: '🍎' },
    { name: 'Diagnoza', href: '/diagnosis', icon: '🧠' },
    { name: 'OCR / Skanuj', href: '/ocr', icon: '📄' },
    { name: 'Raporty', href: '/reports', icon: '📈' },
];

export function Sidebar() {
    const pathname = usePathname();

    return (
        <aside className={styles.sidebar}>
            <div className={styles.logo}>
                <div className={styles.logoIcon} />
                <span className={styles.logoText}>Zdrowie v1</span>
            </div>

            <nav className={styles.nav}>
                {navItems.map((item) => (
                    <Link
                        key={item.href}
                        href={item.href}
                        className={`${styles.navItem} ${pathname === item.href ? styles.navItemActive : ''
                            }`}
                    >
                        <span>{item.icon}</span>
                        <span>{item.name}</span>
                    </Link>
                ))}
            </nav>

            <div className={styles.footer}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                    <button className={styles.navItem} style={{ border: 'none', background: 'none', cursor: 'pointer', flex: 1 }}>
                        <span>⚙️</span>
                        <span>Ustawienia</span>
                    </button>
                    <ThemeToggle />
                </div>
            </div>
        </aside>
    );
}
