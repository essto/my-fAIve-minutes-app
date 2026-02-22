/* apps/web/src/app/(dashboard)/health/page.tsx */
'use client';

import { useEffect, useState } from 'react';
import { HealthChart } from '@/components/shared/charts/HealthChart';
import styles from '@/styles/Feature.module.css';

interface HealthData {
    metrics: {
        heartRate: { current: number; avg7d: number; min: number; max: number };
        sleep: { lastNight: string; avg7d: string; quality: string };
        weight: { current: number; change30d: number; bmi: number };
    };
    charts: {
        heartRateHistory: any;
        sleepHistory: any;
        weightHistory: any;
    };
}

export default function HealthPage() {
    const [data, setData] = useState<HealthData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHealthData = async () => {
            try {
                const token = localStorage.getItem('token');
                const response = await fetch('/api/visualization/health-details', {
                    headers: { Authorization: `Bearer ${token}` },
                });
                if (response.ok) {
                    const result = await response.json();
                    setData(result);
                }
            } catch (err) {
                console.error('Failed to fetch health data:', err);
            } finally {
                setLoading(false);
            }
        };
        fetchHealthData();
    }, []);

    if (loading) {
        return (
            <div className={styles.container}>
                <div className={styles.header}>
                    <div className="h-10 w-48 bg-gray-200 dark:bg-gray-800 animate-pulse rounded-lg" />
                </div>
                <div className={styles.grid}>
                    {[1, 2, 3].map(i => <div key={i} className="h-40 bg-gray-200 dark:bg-gray-800 animate-pulse rounded-2xl" />)}
                </div>
                <div className="h-96 bg-gray-200 dark:bg-gray-800 animate-pulse rounded-2xl mt-8" />
            </div>
        );
    }

    if (!data) return <div className="p-8 text-center text-destructive">Wystąpił błąd podczas ładowania danych medycznych.</div>;

    return (
        <div className={styles.container}>
            <header className={styles.header}>
                <h1 className={styles.title}>Moje Zdrowie</h1>
                <p className={styles.subtitle}>Szczegółowa analiza Twoich parametrów życiowych i trendów.</p>
            </header>

            <section className={styles.grid}>
                <div className={styles.card}>
                    <p className={styles.label}>Tętno spoczynkowe</p>
                    <p className="text-3xl font-black">{data.metrics.heartRate.current} <span className="text-sm font-normal text-color-gray-500">BPM</span></p>
                    <div className="mt-4 flex flex-col gap-1 text-xs text-color-gray-500">
                        <div className="flex justify-between"><span>Średnia (7 dni)</span><span className="font-bold">{data.metrics.heartRate.avg7d}</span></div>
                        <div className="flex justify-between"><span>Min / Max</span><span className="font-bold">{data.metrics.heartRate.min} / {data.metrics.heartRate.max}</span></div>
                    </div>
                </div>

                <div className={styles.card}>
                    <p className={styles.label}>Ostatni Sen</p>
                    <p className="text-3xl font-black">{data.metrics.sleep.lastNight}</p>
                    <div className="mt-4 flex flex-col gap-1 text-xs text-color-gray-500">
                        <div className="flex justify-between"><span>Średnia (7 dni)</span><span className="font-bold">{data.metrics.sleep.avg7d}</span></div>
                        <div className="flex justify-between"><span>Jakość</span><span className="font-bold text-success">{data.metrics.sleep.quality}</span></div>
                    </div>
                </div>

                <div className={styles.card}>
                    <p className={styles.label}>Aktualna Waga</p>
                    <p className="text-3xl font-black">{data.metrics.weight.current} <span className="text-sm font-normal text-color-gray-500">kg</span></p>
                    <div className="mt-4 flex flex-col gap-1 text-xs text-color-gray-500">
                        <div className="flex justify-between"><span>Zmiana (30 dni)</span><span className={data.metrics.weight.change30d <= 0 ? 'text-success font-bold' : 'text-destructive font-bold'}>{data.metrics.weight.change30d} kg</span></div>
                        <div className="flex justify-between"><span>BMI</span><span className="font-bold">{(data.metrics.weight.bmi).toFixed(1)}</span></div>
                    </div>
                </div>
            </section>

            <section className="space-y-8 mt-4">
                <div className={styles.card}>
                    <h2 className={styles.sectionTitle}>Historia Tętna</h2>
                    <HealthChart config={data.charts.heartRateHistory} height={350} />
                </div>

                <div className={styles.card}>
                    <h2 className={styles.sectionTitle}>Trend Wagi (Ostatnie 3 miesiące)</h2>
                    <HealthChart config={data.charts.weightHistory} height={350} />
                </div>

                <div className={styles.card}>
                    <h2 className={styles.sectionTitle}>Analiza Snu (Godziny)</h2>
                    <HealthChart config={data.charts.sleepHistory} height={350} />
                </div>
            </section>
        </div>
    );
}
