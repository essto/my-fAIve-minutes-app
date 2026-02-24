'use client';

import { motion } from 'framer-motion';
import { useTranslations, useLocale } from 'next-intl';
import { useRouter, usePathname } from '@/i18n/routing';
import { useTheme } from 'next-themes';
import { User, Globe, Moon, Bell, Smartphone, Monitor, Sun } from 'lucide-react';
import { useState, useEffect } from 'react';

export default function SettingsPage() {
    const t = useTranslations('Settings');
    const locale = useLocale();
    const router = useRouter();
    const pathname = usePathname();
    const { theme, setTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setMounted(true);
    }, []);

    const handleLanguageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const newLocale = e.target.value as 'pl' | 'en';
        router.replace(pathname, { locale: newLocale });
    };

    return (
        <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500">
            <div>
                <h1 className="text-3xl font-bold tracking-tight text-foreground">{t('title')}</h1>
                <p className="text-muted-foreground mt-1">{t('subtitle')}</p>
            </div>

            <div className="grid gap-6">
                {/* Profile Section */}
                <motion.section
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="glass-card p-6"
                >
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 rounded-lg bg-brand/10 text-brand">
                            <User className="w-5 h-5" />
                        </div>
                        <h2 className="text-xl font-semibold">{t('profile_section')}</h2>
                    </div>
                    <div className="space-y-4 text-sm text-foreground">
                        <div className="grid grid-cols-[1fr_2fr] items-center gap-4 py-3 border-b border-border/50">
                            <span className="text-muted-foreground">{t('name')}</span>
                            <span>Jan Kowalski</span>
                        </div>
                        <div className="grid grid-cols-[1fr_2fr] items-center gap-4 py-3 border-b border-border/50">
                            <span className="text-muted-foreground">{t('email')}</span>
                            <span>jan.kowalski@example.com</span>
                        </div>
                        <button className="text-brand hover:underline font-medium text-sm mt-2">{t('edit_profile')}</button>
                    </div>
                </motion.section>

                {/* Preferences Section: Theme & Language */}
                <div className="grid md:grid-cols-2 gap-6">
                    {/* Theme */}
                    <motion.section
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="glass-card p-6"
                    >
                        <div className="flex items-center gap-3 mb-4">
                            <div className="p-2 rounded-lg bg-status-info/10 text-status-info">
                                <Moon className="w-5 h-5" />
                            </div>
                            <h2 className="text-xl font-semibold">{t('theme_section')}</h2>
                        </div>

                        {mounted && (
                            <div className="flex bg-neutral-bg3 p-1 rounded-xl">
                                <button
                                    aria-label="light_theme"
                                    onClick={() => setTheme('light')}
                                    className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg transition-all text-sm font-medium ${theme === 'light' ? 'bg-white text-black shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                                >
                                    <Sun className="w-4 h-4" /> Light
                                </button>
                                <button
                                    onClick={() => setTheme('dark')}
                                    className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg transition-all text-sm font-medium ${theme === 'dark' ? 'bg-neutral-bg text-white shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                                >
                                    <Moon className="w-4 h-4" /> Dark
                                </button>
                                <button
                                    onClick={() => setTheme('system')}
                                    className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg transition-all text-sm font-medium ${theme === 'system' ? 'bg-neutral-bg text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                                >
                                    <Monitor className="w-4 h-4" /> System
                                </button>
                            </div>
                        )}
                    </motion.section>

                    {/* Language */}
                    <motion.section
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="glass-card p-6"
                    >
                        <div className="flex items-center gap-3 mb-4">
                            <div className="p-2 rounded-lg bg-status-success/10 text-status-success">
                                <Globe className="w-5 h-5" />
                            </div>
                            <h2 className="text-xl font-semibold">{t('language_section')}</h2>
                        </div>
                        <select
                            aria-label="language"
                            value={locale}
                            onChange={handleLanguageChange}
                            className="w-full bg-neutral-bg3 border border-border rounded-xl px-4 py-3 text-foreground focus:ring-2 focus:ring-brand/50 outline-none transition-all cursor-pointer"
                        >
                            <option value="pl">Polski (PL)</option>
                            <option value="en">English (EN)</option>
                        </select>
                    </motion.section>
                </div>

                {/* Notifications */}
                <motion.section
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="glass-card p-6"
                >
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 rounded-lg bg-status-warning/10 text-status-warning">
                            <Bell className="w-5 h-5" />
                        </div>
                        <h2 className="text-xl font-semibold">{t('notifications_section')}</h2>
                    </div>

                    <div className="space-y-4">
                        <label className="flex items-center justify-between cursor-pointer group">
                            <div>
                                <h3 className="text-sm font-medium text-foreground">{t('push_notifications')}</h3>
                                <p className="text-xs text-muted-foreground">{t('push_notifications_desc')}</p>
                            </div>
                            <div className="relative inline-block w-12 h-6 rounded-full bg-brand transition-colors">
                                <span className="absolute left-[26px] top-1 w-4 h-4 rounded-full bg-white transition-transform" />
                            </div>
                        </label>
                        <label className="flex items-center justify-between cursor-pointer group pt-4 border-t border-border/50">
                            <div>
                                <h3 className="text-sm font-medium text-foreground">{t('email_notifications')}</h3>
                                <p className="text-xs text-muted-foreground">{t('email_notifications_desc')}</p>
                            </div>
                            <div className="relative inline-block w-12 h-6 rounded-full bg-neutral-bg3 transition-colors border border-border">
                                <span className="absolute left-1 top-[3px] w-4 h-4 rounded-full bg-muted-foreground transition-transform" />
                            </div>
                        </label>
                    </div>
                </motion.section>

                {/* Devices */}
                <motion.section
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="glass-card p-6"
                >
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
                            <Smartphone className="w-5 h-5" />
                        </div>
                        <h2 className="text-xl font-semibold">{t('devices_section')}</h2>
                    </div>

                    <div className="flex flex-col sm:flex-row items-center justify-between p-4 bg-neutral-bg2 rounded-xl border border-border/50 gap-4">
                        <div className="flex items-center gap-4 text-left w-full sm:w-auto">
                            <div className="w-10 h-10 bg-black rounded-full flex items-center justify-center text-white border border-white/10 shrink-0">
                                🍏
                            </div>
                            <div>
                                <h3 className="text-sm font-semibold text-foreground">Apple Health</h3>
                                <p className="text-xs text-status-success">{t('connected')}</p>
                            </div>
                        </div>
                        <button className="text-sm px-4 py-2 border border-border hover:bg-neutral-bg3 rounded-lg transition-colors w-full sm:w-auto">
                            {t('manage')}
                        </button>
                    </div>
                </motion.section>
            </div>
        </div>
    );
}
