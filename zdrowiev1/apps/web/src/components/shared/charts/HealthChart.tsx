/* apps/web/src/components/shared/charts/HealthChart.tsx */
'use client';

import React from 'react';
import {
    LineChart, Line, AreaChart, Area, BarChart, Bar,
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';

export interface ChartDataPoint {
    label: string;
    value: number;
    [key: string]: unknown;
}

export interface ChartConfig {
    type: 'line' | 'area' | 'bar' | 'radar' | 'gauge' | 'heatmap' | 'scatter' | 'progress_ring' | 'sparkline' | 'candlestick';
    title?: string;
    data: ChartDataPoint[];
    colors: string[];
    xAxisKey?: string;
    yAxisKey?: string;
}

interface HealthChartProps {
    config: ChartConfig;
    height?: number | string;
}

export function HealthChart({ config, height = 300 }: HealthChartProps) {
    const { type, data, colors, xAxisKey = 'label', yAxisKey = 'value' } = config;

    const renderChart = () => {
        switch (type) {
            case 'line':
            case 'sparkline':
                return (
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                        <XAxis dataKey={xAxisKey} stroke="var(--color-gray-400)" fontSize={12} tickLine={false} axisLine={false} />
                        <YAxis stroke="var(--color-gray-400)" fontSize={12} tickLine={false} axisLine={false} />
                        <Tooltip
                            contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}
                            itemStyle={{ color: 'var(--foreground)' }}
                        />
                        <Line
                            type="monotone"
                            dataKey={yAxisKey}
                            stroke={colors[0]}
                            strokeWidth={3}
                            dot={{ fill: colors[0], strokeWidth: 2, r: 4 }}
                            activeDot={{ r: 6, strokeWidth: 0 }}
                        />
                    </LineChart>
                );

            case 'area':
                return (
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id="colorArea" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={colors[0]} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={colors[0]} stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                        <XAxis dataKey={xAxisKey} stroke="var(--color-gray-400)" fontSize={12} tickLine={false} axisLine={false} />
                        <YAxis stroke="var(--color-gray-400)" fontSize={12} tickLine={false} axisLine={false} />
                        <Tooltip
                            contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}
                        />
                        <Area type="monotone" dataKey={yAxisKey} stroke={colors[0]} fillOpacity={1} fill="url(#colorArea)" />
                    </AreaChart>
                );

            case 'bar':
                return (
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                        <XAxis dataKey={xAxisKey} stroke="var(--color-gray-400)" fontSize={12} tickLine={false} axisLine={false} />
                        <YAxis stroke="var(--color-gray-400)" fontSize={12} tickLine={false} axisLine={false} />
                        <Tooltip
                            contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}
                        />
                        <Bar dataKey={yAxisKey} fill={colors[0]} radius={[4, 4, 0, 0]} />
                    </BarChart>
                );

            case 'radar':
                return (
                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
                        <PolarGrid stroke="var(--border)" />
                        <PolarAngleAxis dataKey={xAxisKey} stroke="var(--color-gray-400)" fontSize={10} />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="var(--color-gray-400)" fontSize={10} />
                        <Radar
                            name="Wartość"
                            dataKey={yAxisKey}
                            stroke={colors[0]}
                            fill={colors[0]}
                            fillOpacity={0.6}
                        />
                        <Tooltip
                            contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}
                        />
                    </RadarChart>
                );

            default:
                return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--color-gray-400)' }}>Typ wykresu {type} nie jest jeszcze obsługiwany</div>;
        }
    };

    return (
        <div style={{ width: '100%', height }}>
            <ResponsiveContainer width="100%" height="100%">
                {renderChart()}
            </ResponsiveContainer>
        </div>
    );
}
