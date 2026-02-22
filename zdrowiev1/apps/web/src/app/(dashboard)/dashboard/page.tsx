'use client';

import { useEffect, useState } from 'react';
import { HealthChart } from '@/components/shared/charts/HealthChart';
import { NotificationBell } from './NotificationBell';
import styles from './Dashboard.module.css';

interface Anomaly {
    metric: string;
    value: number;
    severity: 'low' | 'medium' | 'high';
    message: string;
}

interface ChartConfig {
    type: 'line' | 'area' | 'bar' | 'radar' | 'gauge' | 'heatmap' | 'scatter' | 'progress_ring' | 'sparkline' | 'candlestick';
    title?: string;
    data: any[];
    colors: string[];
}

interface DashboardData {
    healthScore: number;
    anomalies: Anomaly[];
    charts: {
        healthTrend: ChartConfig;
        dietTrend: ChartConfig;
        activityRings: ChartConfig;
        heartRate: ChartConfig;
    };
}

export default function Dashboard() {
    const [data, setData] = useState<DashboardData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const token = localStorage.getItem('token');
                const response = await fetch('/api/visualization/dashboard', {
                    headers: { Authorization: `Bearer ${token}` },
                });

                if (!response.ok) throw new Error('Błąd pobierania danych');

                const result = await response.json();
                setData(result);
            } catch (err: any) {
                console.error('Failed to fetch dashboard data:', err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    if (loading) {
        return (
            <div className={styles.dashboard}>
                <div className={styles.header}>
                    <div className={`${styles.skeleton} h-10 w-48`} />
                    <div className={`${styles.skeleton} h-10 w-10 rounded-full`} />
                </div>
                <div className={styles.statsGrid}>
                    <div className={`${styles.skeleton} h-32`} />
                    <div className={`${styles.skeleton} h-32`} />
                    <div className={`${styles.skeleton} h-32`} />
                </div>
                <div className={styles.chartsGrid}>
                    <div className={`${styles.skeleton} h-64`} />
                    <div className={`${styles.skeleton} h-64`} />
                </div>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className={`${styles.card} flex flex-col items-center justify-center h-96 text-center`}>
                <p className="text-xl font-semibold text-destructive mb-2">Ups! Coś poszło nie tak.</p>
                <p className="text-color-gray-500 mb-6">{error || 'Nie udało się załadować danych'}</p>
                <button
                    onClick={() => window.location.reload()}
                    className="px-6 py-2 bg-primary text-white rounded-lg"
                >
                    Odśwież
                </button>
            </div>
        );
    }

    return (
        <div className={styles.dashboard}>
            <header className={styles.header}>
                <h1 className={styles.title}>Panel Zdrowia</h1>
                <div className="flex items-center gap-4">
                    <NotificationBell />
                </div>
            </header>

            <section className={styles.statsGrid}>
                {/* Health Score Card */}
                <div className={styles.card}>
                    <h2 className={styles.cardTitle}>Global Health Score</h2>
                    <div className={styles.scoreContainer}>
                        <div className={styles.scoreCircle}>
                            <svg viewBox="0 0 100 100" className="w-full h-full">
                                <circle
                                    cx="50" cy="50" r="45"
                                    fill="none"
                                    stroke="var(--border)"
                                    strokeWidth="8"
                                />
                                <circle
                                    cx="50" cy="50" r="45"
                                    fill="none"
                                    stroke="var(--primary)"
                                    strokeWidth="8"
                                    strokeDasharray="283"
                                    strokeDashoffset={283 - (283 * data.healthScore) / 100}
                                    strokeLinecap="round"
                                    transform="rotate(-90 50 50)"
                                    style={{ transition: 'stroke-dashoffset 1s ease-out' }}
                                />
                            </svg>
                            <span className={styles.scoreValue}>{data.healthScore}</span>
                        </div>
                        <div>
                            <p className="text-sm text-color-gray-500">Twój aktualny wskaźnik zdrowia obliczony na podstawie tętna, snu i diety.</p>
                        </div>
                    </div>
                </div>

                {/* Anomalies Card */}
                <div className={`${styles.card} ${data.anomalies.length > 0 ? styles.anomalyCard : ''}`}>
                    <h2 className={styles.cardTitle}>Wykryte Anomalie</h2>
                    {data.anomalies.length === 0 ? (
                        <p className="text-success flex items-center gap-2">
                            <span>✅</span> Wszystkie parametry w normie
                        </p>
                    ) : (
                        <ul className={styles.anomalyList}>
                            {data.anomalies.map((anomaly, idx) => (
                                <li key={idx} className={styles.anomalyItem}>
                                    <span>⚠️</span>
                                    <div>
                                        <p className={anomaly.severity === 'high' ? styles.severityHigh : styles.severityMedium}>
                                            {anomaly.message}
                                        </p>
                                        <p className="text-xs opacity-70">Wartość: {anomaly.value}</p>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>

                {/* Summary Card */}
                <div className={styles.card}>
                    <h2 className={styles.cardTitle}>Ostatnia Aktywność</h2>
                    <div className="flex flex-col gap-2">
                        <div className="flex justify-between items-center text-sm">
                            <span>Waga</span>
                            <span className="font-bold">72.5 kg</span>
                        </div>
                        <div className="flex justify-between items-center text-sm">
                            <span>Kalorie dziś</span>
                            <span className="font-bold">1,850 kcal</span>
                        </div>
                        <div className="flex justify-between items-center text-sm">
                            <span>Sen</span>
                            <span className="font-bold">7h 20m</span>
                        </div>
                    </div>
                </div>
            </section>

            <section className={styles.chartsGrid}>
                <div className={styles.chartCard}>
                    <h2 className={styles.cardTitle}>{data.charts.healthTrend.title || 'Trend Zdrowia'}</h2>
                    <HealthChart config={data.charts.healthTrend} />
                </div>
                <div className={styles.chartCard}>
                    <h2 className={styles.cardTitle}>{data.charts.heartRate.title || 'Tętno'}</h2>
                    <HealthChart config={data.charts.heartRate} />
                </div>
                <div className={styles.chartCard}>
                    <h2 className={styles.cardTitle}>{data.charts.dietTrend.title || 'Bilans Kaloryczny'}</h2>
                    <HealthChart config={data.charts.dietTrend} />
                </div>
                <div className={styles.chartCard}>
                    <h2 className={styles.cardTitle}>Podsumowanie Aktywności</h2>
                    <HealthChart config={data.charts.activityRings} />
                </div>
            </section>
        </div>
    );
}
