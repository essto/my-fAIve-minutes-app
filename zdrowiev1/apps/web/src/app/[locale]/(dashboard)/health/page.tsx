/* apps/web/src/app/(dashboard)/health/page.tsx */
'use client';

import { useEffect, useState } from 'react';
import { HealthChart, type ChartConfig } from '@/components/shared/charts/HealthChart';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/shared/ui/Card/Card';
import { SkeletonLoader } from '@/components/shared/ui/SkeletonLoader/SkeletonLoader';
import { Button } from '@/components/shared/ui/Button/Button';
import styles from '@/styles/Feature.module.css';

interface HealthData {
    metrics: {
        heartRate: { current: number; avg7d: number; min: number; max: number };
        sleep: { lastNight: string; avg7d: string; quality: string };
        weight: { current: number; change30d: number; bmi: number };
    };
    charts: {
        heartRateHistory: ChartConfig;
        sleepHistory: ChartConfig;
        weightHistory: ChartConfig;
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
                <header className={styles.header}>
                    <SkeletonLoader className="h-10 w-48 mb-2" />
                    <SkeletonLoader className="h-4 w-64" />
                </header>
                <div className={styles.grid}>
                    {[1, 2, 3].map(i => (
                        <Card key={i}>
                            <CardContent className="h-40 flex items-center justify-center">
                                <SkeletonLoader className="h-full w-full" />
                            </CardContent>
                        </Card>
                    ))}
                </div>
                <Card className="mt-8">
                    <CardContent className="h-96">
                        <SkeletonLoader className="h-full w-full" />
                    </CardContent>
                </Card>
            </div>
        );
    }

    if (!data) return <div className="p-8 text-center text-destructive">Wystąpił błąd podczas ładowania danych medycznych.</div>;

    return (
        <div className={styles.container}>
            <header className={styles.header}>
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className={styles.title}>Moje Zdrowie</h1>
                        <p className={styles.subtitle}>Szczegółowa analiza Twoich parametrów życiowych i trendów.</p>
                    </div>
                    <Button variant="secondary" onClick={() => window.print()}>
                        Pobierz raport
                    </Button>
                </div>
            </header>

            <section className={styles.grid}>
                <Card interactive>
                    <CardContent className="pt-6">
                        <p className={styles.label}>Tętno spoczynkowe</p>
                        <p className="text-3xl font-black">{data.metrics.heartRate.current} <span className="text-sm font-normal text-color-gray-500">BPM</span></p>
                        <div className="mt-4 flex flex-col gap-1 text-xs text-color-gray-500">
                            <div className="flex justify-between"><span>Średnia (7 dni)</span><span className="font-bold">{data.metrics.heartRate.avg7d}</span></div>
                            <div className="flex justify-between"><span>Min / Max</span><span className="font-bold">{data.metrics.heartRate.min} / {data.metrics.heartRate.max}</span></div>
                        </div>
                    </CardContent>
                </Card>

                <Card interactive>
                    <CardContent className="pt-6">
                        <p className={styles.label}>Ostatni Sen</p>
                        <p className="text-3xl font-black">{data.metrics.sleep.lastNight}</p>
                        <div className="mt-4 flex flex-col gap-1 text-xs text-color-gray-500">
                            <div className="flex justify-between"><span>Średnia (7 dni)</span><span className="font-bold">{data.metrics.sleep.avg7d}</span></div>
                            <div className="flex justify-between"><span>Jakość</span><span className="font-bold text-success">{data.metrics.sleep.quality}</span></div>
                        </div>
                    </CardContent>
                </Card>

                <Card interactive>
                    <CardContent className="pt-6">
                        <p className={styles.label}>Aktualna Waga</p>
                        <p className="text-3xl font-black">{data.metrics.weight.current} <span className="text-sm font-normal text-color-gray-500">kg</span></p>
                        <div className="mt-4 flex flex-col gap-1 text-xs text-color-gray-500">
                            <div className="flex justify-between"><span>Zmiana (30 dni)</span><span className={data.metrics.weight.change30d <= 0 ? 'text-success font-bold' : 'text-destructive font-bold'}>{data.metrics.weight.change30d} kg</span></div>
                            <div className="flex justify-between"><span>BMI</span><span className="font-bold">{(data.metrics.weight.bmi).toFixed(1)}</span></div>
                        </div>
                    </CardContent>
                </Card>
            </section>

            <section className="space-y-8 mt-4">
                <Card>
                    <CardHeader>
                        <CardTitle>Historia Tętna</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <HealthChart config={data.charts.heartRateHistory} height={350} />
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle>Trend Wagi (Ostatnie 3 miesiące)</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <HealthChart config={data.charts.weightHistory} height={350} />
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle>Analiza Snu (Godziny)</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <HealthChart config={data.charts.sleepHistory} height={350} />
                    </CardContent>
                </Card>
            </section>
        </div>
    );
}
