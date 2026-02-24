'use client';

import { useState, useEffect, useRef } from 'react';

interface Notification {
    id: string;
    type: 'SYSTEM' | 'ANOMALY' | 'REMINDER';
    title: string;
    message: string;
    channel: 'IN_APP' | 'EMAIL' | 'PUSH';
    isRead: boolean;
    createdAt: string;
}

export function NotificationBell() {
    const [notifications, setNotifications] = useState<Notification[]>([]);
    const [open, setOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    const fetchNotifications = async () => {
        try {
            const response = await fetch('/api/notifications');
            if (response.ok) {
                const data = await response.ok ? await response.json() : [];
                setNotifications(Array.isArray(data) ? data : []);
            }
        } catch (error) {
            console.error('Błąd pobierania powiadomień:', error);
        }
    };

    useEffect(() => {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        fetchNotifications();
        const interval = setInterval(fetchNotifications, 30000);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setOpen(false);
            }
        }
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const markAsRead = async (id: string) => {
        try {
            const response = await fetch(`/api/notifications/${id}/read`, {
                method: 'PATCH',
            });
            if (response.ok) {
                setNotifications((prev) =>
                    prev.map((n) => (n.id === id ? { ...n, isRead: true } : n))
                );
            }
        } catch (error) {
            console.error('Błąd oznaczania jako przeczytane:', error);
        }
    };

    const unreadCount = notifications.filter((n) => !n.isRead).length;

    return (
        <div className="relative" ref={dropdownRef} data-testid="notification-bell">
            <button
                id="notification-bell-button"
                aria-label="Powiadomienia"
                onClick={() => setOpen(!open)}
                className="relative p-2 rounded-full hover:bg-gray-100 dark:hover:bg-slate-700 transition"
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6 text-slate-600 dark:text-slate-300"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
                    />
                </svg>
                {unreadCount > 0 && (
                    <span
                        data-testid="notification-badge"
                        className="absolute top-0 right-0 bg-red-500 text-white text-[10px] rounded-full h-4 w-4 flex items-center justify-center font-bold"
                    >
                        {unreadCount}
                    </span>
                )}
            </button>

            {open && (
                <div
                    data-testid="notification-panel"
                    className="absolute right-0 mt-2 w-80 bg-card rounded-xl shadow-glow border border-border z-50 overflow-hidden"
                >
                    <div className="p-4 border-b border-border flex justify-between items-center">
                        <h3 className="font-bold text-foreground">Powiadomienia</h3>
                        {unreadCount > 0 && (
                            <span className="text-xs bg-brand/10 text-brand px-2 py-1 rounded-full font-medium">
                                {unreadCount} nowe
                            </span>
                        )}
                    </div>
                    <div className="max-h-96 overflow-y-auto">
                        {notifications.length === 0 ? (
                            <div className="p-8 text-center text-muted-foreground text-sm">Brak powiadomień</div>
                        ) : (
                            notifications.map((n) => (
                                <div
                                    key={n.id}
                                    onClick={() => !n.isRead && markAsRead(n.id)}
                                    className={`p-4 border-b border-border hover:bg-neutral-bg3 cursor-pointer transition-colors ${!n.isRead ? 'bg-brand/5' : ''
                                        }`}
                                >
                                    <div className="flex gap-3">
                                        <div
                                            className={`h-2 w-2 rounded-full mt-2 shrink-0 ${n.type === 'ANOMALY' ? 'bg-destructive' : 'bg-brand'
                                                } ${n.isRead ? 'opacity-0' : 'opacity-100'}`}
                                        />
                                        <div>
                                            <p className="text-sm font-medium text-foreground">
                                                {n.title}
                                            </p>
                                            <p className="text-xs text-muted-foreground mt-1">
                                                {n.message}
                                            </p>
                                            <p className="text-[10px] text-muted-foreground/60 mt-2">
                                                {new Date(n.createdAt).toLocaleString('pl-PL')}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                    <div className="p-3 bg-neutral-bg2 hover:bg-neutral-bg3 text-center border-t border-border transition-colors cursor-pointer">
                        <button className="text-xs text-brand hover:text-brand-light font-medium">
                            Zobacz wszystkie
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
