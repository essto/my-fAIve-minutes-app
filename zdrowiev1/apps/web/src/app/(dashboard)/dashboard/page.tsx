'use client';

import { useEffect, useState } from 'react';
import { HealthChart } from '@/components/shared/charts/HealthChart';
import { NotificationBell } from './NotificationBell';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/shared/ui/Card/Card';
import { SkeletonLoader } from '@/components/shared/ui/SkeletonLoader/SkeletonLoader';
import { motion, Variants } from 'framer-motion';

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

const containerVariants: Variants = {
    hidden: { opacity: 0 },
    show: {
        opacity: 1,
        transition: { staggerChildren: 0.1 }
    }
};

const itemVariants: Variants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

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
            <div className="flex flex-col gap-6">
                <header className="flex flex-col gap-2">
                    <SkeletonLoader className="h-10 w-64 rounded-md" />
                    <SkeletonLoader className="h-5 w-96 rounded-md" />
                </header>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-4">
                    <Card className="glass-card col-span-1 lg:col-span-1">
                        <CardContent className="flex justify-center items-center h-48 pt-6">
                            <SkeletonLoader variant="circle" className="w-32 h-32" />
                        </CardContent>
                    </Card>
                    <Card className="glass-card col-span-1 md:col-span-2 lg:col-span-2">
                        <CardHeader>
                            <SkeletonLoader className="h-6 w-32 rounded-md" />
                        </CardHeader>
                        <CardContent>
                            <SkeletonLoader className="h-[120px] w-full rounded-md" />
                        </CardContent>
                    </Card>
                    <Card className="glass-card col-span-1 md:col-span-1 lg:col-span-1">
                        <CardHeader>
                            <SkeletonLoader className="h-6 w-32 rounded-md" />
                        </CardHeader>
                        <CardContent>
                            <SkeletonLoader className="h-4 w-full mb-2 rounded-md" />
                            <SkeletonLoader className="h-4 w-3/4 rounded-md" />
                        </CardContent>
                    </Card>
                </div>
            </div>
        );
    }

    if (error) {
        return <div className="p-6 bg-destructive/10 text-destructive rounded-xl border border-destructive/20">{error}</div>;
    }

    if (!data) return null;

    return (
        <div className="flex flex-col gap-6">
            <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-foreground">Witaj ponownie! 👋</h1>
                    <p className="text-muted-foreground mt-1 text-sm md:text-base">Oto podsumowanie Twojego zdrowia na dziś</p>
                </div>
                <div className="flex items-center">
                    <NotificationBell />
                </div>
            </header>

            <motion.div
                className="grid grid-cols-1 lg:grid-cols-3 gap-6"
                variants={containerVariants}
                initial="hidden"
                animate="show"
            >
                {/* Główny wynik zdrowia */}
                <motion.div variants={itemVariants} className="col-span-1">
                    <Card className="glass-card relative overflow-hidden h-full flex flex-col justify-center border-brand/20 bg-gradient-to-br from-brand/5 to-transparent">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-brand to-brand-light"></div>
                        <CardContent className="pt-6 flex flex-col items-center justify-center gap-6">
                            <div className="relative w-36 h-36 flex items-center justify-center">
                                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                                    <circle cx="50" cy="50" r="45" className="fill-none stroke-neutral-bg3 stroke-[8px]" />
                                    <motion.circle
                                        cx="50" cy="50" r="45"
                                        className="fill-none stroke-brand stroke-[8px]"
                                        initial={{ strokeDasharray: "0 283" }}
                                        animate={{ strokeDasharray: `${(data.healthScore / 100) * 283} 283` }}
                                        transition={{ duration: 1.5, ease: "easeOut" }}
                                        strokeLinecap="round"
                                    />
                                </svg>
                                <div className="absolute flex items-baseline gap-0.5">
                                    <span className="text-5xl font-bold tracking-tighter text-foreground">{data.healthScore}</span>
                                    <span className="text-sm font-medium text-muted-foreground">/100</span>
                                </div>
                            </div>
                            <div className="text-center">
                                <h3 className="text-lg font-semibold text-foreground">Twój wynik zdrowia</h3>
                                <p className="text-sm text-muted-foreground mt-1 px-4">Utrzymujesz się w górnych 20% w swojej grupie wiekowej.</p>
                            </div>
                        </CardContent>
                    </Card>
                </motion.div>

                {/* Wykryte anomalia */}
                <motion.div variants={itemVariants} className="col-span-1 lg:col-span-2">
                    <Card className="glass-card h-full">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-status-warning"></span>
                                Wymaga uwagi
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            {data.anomalies.length > 0 ? (
                                <ul className="flex flex-col gap-3 m-0 p-0 list-none">
                                    {data.anomalies.map((anomaly, index) => {
                                        const isHigh = anomaly.severity === 'high';
                                        return (
                                            <li key={index} className={`flex items-start gap-4 p-4 rounded-xl border ${isHigh ? 'bg-destructive/10 border-destructive/20' : 'bg-status-warning/10 border-status-warning/20'} transition-all hover:scale-[1.01]`}>
                                                <div className={`mt-0.5 ${isHigh ? 'text-destructive' : 'text-status-warning'}`}>
                                                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" /><path d="M12 9v4" /><path d="M12 17h.01" /></svg>
                                                </div>
                                                <div className="flex flex-col">
                                                    <span className="font-medium text-foreground">{anomaly.metric}</span>
                                                    <span className="text-sm opacity-90 mt-0.5">{anomaly.message}</span>
                                                </div>
                                            </li>
                                        );
                                    })}
                                </ul>
                            ) : (
                                <div className="flex flex-col items-center justify-center h-32 gap-3 text-center">
                                    <div className="w-12 h-12 rounded-full bg-status-success/20 text-status-success flex items-center justify-center">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><path d="m9 11 3 3L22 4" /></svg>
                                    </div>
                                    <p className="text-muted-foreground font-medium">Wszystkie wskaźniki w normie!</p>
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </motion.div>

                {/* Trendy zdrowotne */}
                <motion.div variants={itemVariants} className="col-span-1 lg:col-span-2">
                    <Card className="glass-card">
                        <CardHeader>
                            <CardTitle>Trendy zdrowotne (30 dni)</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="h-[250px] w-full" data-testid="weight-chart">
                                <HealthChart config={data.charts.healthTrend} />
                            </div>
                        </CardContent>
                    </Card>
                </motion.div>

                {/* Aktywność i sen */}
                <motion.div variants={itemVariants} className="col-span-1">
                    <Card className="glass-card h-full">
                        <CardHeader>
                            <CardTitle>Aktywność i regeneracja</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="h-[250px] w-full" data-testid="activity-rings">
                                <HealthChart config={data.charts.activityRings} />
                            </div>
                        </CardContent>
                    </Card>
                </motion.div>
            </motion.div>
        </div>
    );
}
