/* apps/web/src/components/shared/layout/AppShell.tsx */
'use client';

import React from 'react';
import { Sidebar } from './Sidebar';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ThemeToggle } from '../ui/ThemeToggle/ThemeToggle';
import { motion } from 'framer-motion';

interface AppShellProps {
    children: React.ReactNode;
}

const mobileNavItems = [
    { name: 'Dashboard', href: '/dashboard', icon: <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="7" height="9" x="3" y="3" rx="1" /><rect width="7" height="5" x="14" y="3" rx="1" /><rect width="7" height="9" x="14" y="12" rx="1" /><rect width="7" height="5" x="3" y="16" rx="1" /></svg> },
    { name: 'Zdrowie', href: '/health', icon: <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" /></svg> },
    { name: 'Dieta', href: '/diet', icon: <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z" /><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" /><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" /><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" /></svg> },
    { name: 'Diagnoza', href: '/diagnosis', icon: <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z" /><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z" /></svg> },
];

export function AppShell({ children }: AppShellProps) {
    const pathname = usePathname();

    return (
        <div className="flex min-h-screen bg-background">
            <Sidebar />

            <main className="flex-1 md:ml-[260px] p-4 md:p-8 pt-20 md:pt-8 pb-24 md:pb-8 transition-all duration-300">
                <motion.div
                    key={pathname}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                >
                    {children}
                </motion.div>
            </main>

            {/* Mobile Top Header (Glassmorphism) */}
            <header className="md:hidden fixed top-0 w-full flex items-center justify-between p-4 glass z-50 border-b border-border">
                <div className="flex items-center gap-2 text-brand">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v20" /><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" /></svg>
                    <h1 className="text-lg font-bold text-foreground tracking-tight m-0">Zdrowie App</h1>
                </div>
                <ThemeToggle />
            </header>

            {/* Mobile Bottom Navigation (Glassmorphism) */}
            <nav className="md:hidden fixed bottom-0 w-full glass border-t border-border z-50" style={{ paddingBottom: 'env(safe-area-inset-bottom)' }} data-testid="mobile-bottom-nav">
                <div className="flex justify-around items-center h-16 px-2">
                    {mobileNavItems.map((item) => {
                        const isActive = pathname.startsWith(item.href);
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={`flex flex-col items-center justify-center gap-1 flex-1 h-full transition-colors duration-200 ${isActive ? 'text-brand' : 'text-muted-foreground hover:text-foreground'}`}
                            >
                                <span className={`transition-transform duration-200 ${isActive ? '-translate-y-0.5' : ''}`}>
                                    {item.icon}
                                </span>
                                <span className="text-[10px] font-medium">{item.name}</span>
                            </Link>
                        );
                    })}
                </div>
            </nav>
        </div>
    );
}
