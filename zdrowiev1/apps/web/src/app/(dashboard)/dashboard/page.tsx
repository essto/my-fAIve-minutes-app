'use client';

import { useEffect, useState } from 'react';
import { HealthChart } from '@/components/shared/charts/HealthChart';
import { NotificationBell } from './NotificationBell';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/shared/ui/Card/Card';
import { SkeletonLoader } from '@/components/shared/ui/SkeletonLoader/SkeletonLoader';
import styles from './Dashboard.module.css';

interface Anomaly {
    metric: string;
    value: number;
    severity: 'low' | 'medium' | 'high';
    message: string;
}

interface ChartConfig {
    type: 'line' | 'area' | 'bar' | 'radar' | 'gauge' | 'heatmap' | 'scatter' | 'progress_ring' | 'sparkline' | 'candlestick';
    data: any[];
    keys: string[];
    colors: string[];
    indexBy?: string;
}

interface DashboardData {
    healthScore: number;
    anomalies: Anomaly[];
    charts: {
        healthTrend: ChartConfig;
        activityRings: ChartConfig;
        sleepQuality: ChartConfig;
    };
}

export default function Dashboard() {
    const [data, setData] = useState<DashboardData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchDashboardData = async () => {
            try {
                const response = await fetch('/api/dashboard');
                if (!response.ok) throw new Error('Błąd pobierania danych');
                const result = await response.json();
                setData(result);
            } catch (err: any) {
                setError(err.message || 'Wystąpił błąd');
            } finally {
                setLoading(false);
            }
        };

        fetchDashboardData();
    }, []);

    if (loading) {
        return (
            <div className={styles.dashboard}>
                <header className={styles.header}>
                    <div>
                        <SkeletonLoader className="h-8 w-48 mb-2" />
                        <SkeletonLoader className="h-4 w-64" />
                    </div>
                </header>
                <div className={styles.grid}>
                    {/* Main Health Score Skeleton */}
                    <Card className={styles.mainScoreCard}>
                        <CardContent className="flex justify-center items-center h-48">
                            <SkeletonLoader variant="circle" className="w-32 h-32" />
                        </CardContent>
                    </Card>

                    {/* Secondary Cards Skeletons */}
                    <Card>
                        <CardHeader>
                            <SkeletonLoader className="h-6 w-32" />
                        </CardHeader>
                        <CardContent>
                            <SkeletonLoader className="h-32 w-full" />
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <SkeletonLoader className="h-6 w-32" />
                        </CardHeader>
                        <CardContent>
                            <SkeletonLoader className="h-4 w-full mb-2" />
                            <SkeletonLoader className="h-4 w-3/4" />
                        </CardContent>
                    </Card>
                </div>
            </div>
        );
    }

    if (error) {
        return <div className={styles.error}>{error}</div>;
    }

    if (!data) return null;

    return (
        <div className={styles.dashboard}>
            <header className={styles.header}>
                <div>
                    <h1 className={styles.title}>Witaj ponownie! 👋</h1>
                    <p className={styles.subtitle}>Oto podsumowanie Twojego zdrowia na dziś</p>
                </div>
                <div className={styles.actions}>
                    <NotificationBell />
                </div>
            </header>

            <div className={styles.grid}>
                {/* Główny wynik zdrowia */}
                <Card className={styles.mainScoreCard} gradientAccent>
                    <CardContent className={styles.scoreContent}>
                        <div className={styles.scoreRingContainer}>
                            <svg className={styles.scoreRing} viewBox="0 0 100 100">
                                <circle cx="50" cy="50" r="45" className={styles.scoreRingBg} />
                                <circle
                                    cx="50" cy="50" r="45"
                                    className={styles.scoreRingProgress}
                                    strokeDasharray={`${(data.healthScore / 100) * 283} 283`}
                                />
                            </svg>
                            <div className={styles.scoreValue}>
                                <span>{data.healthScore}</span>
                                <span className={styles.scoreLabel}>/100</span>
                            </div>
                        </div>
                        <div className={styles.scoreInfo}>
                            <h3>Twój wynik zdrowia</h3>
                            <p>Utrzymujesz się w górnych 20% w swojej grupie wiekowej.</p>
                        </div>
                    </CardContent>
                </Card>

                {/* Wykryte anomalia */}
                <Card className={styles.anomaliesCard}>
                    <CardHeader>
                        <CardTitle>Wymaga uwagi</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {data.anomalies.length > 0 ? (
                            <ul className={styles.anomaliesList}>
                                {data.anomalies.map((anomaly, index) => (
                                    <li key={index} className={`${styles.anomalyItem} ${styles[anomaly.severity]}`}>
                                        <div className={styles.anomalyIcon}>⚠️</div>
                                        <div className={styles.anomalyDetails}>
                                            <span className={styles.anomalyMetric}>{anomaly.metric}</span>
                                            <span className={styles.anomalyMessage}>{anomaly.message}</span>
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <div className={styles.noAnomalies}>
                                <span className={styles.successIcon}>✅</span>
                                <p>Wszystkie wskaźniki w normie!</p>
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* Trendy zdrowotne */}
                <Card className={styles.chartCard}>
                    <CardHeader>
                        <CardTitle>Trendy zdrowotne (30 dni)</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className={styles.chartContainer} data-testid="weight-chart">
                            <HealthChart config={data.charts.healthTrend} />
                        </div>
                    </CardContent>
                </Card>

                {/* Aktywność i sen */}
                <Card className={styles.chartCard}>
                    <CardHeader>
                        <CardTitle>Aktywność i regeneracja</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className={styles.chartContainer} data-testid="activity-rings">
                            <HealthChart config={data.charts.activityRings} />
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
