'use client'
import { useEffect, useState } from 'react'
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip } from 'recharts'
import { NotificationBell } from './NotificationBell'

interface DashboardData {
    healthScore: number
    anomalies: string[]
    charts: {
        weight: { date: string; value: number }[]
        activityRings: { move: number; sleep: number; diet: number }
    }
}

function ActivityRings({ move, sleep, diet }: { move: number; sleep: number; diet: number }) {
    const rings = [
        { label: 'Ruch', value: move, color: '#ef4444' },
        { label: 'Sen', value: sleep, color: '#6366f1' },
        { label: 'Dieta', value: diet, color: '#22c55e' },
    ]
    const r = 30; const c = 2 * Math.PI * r
    return (
        <div className="flex justify-around" data-testid="activity-rings">
            {rings.map(({ label, value, color }) => (
                <div key={label} className="flex flex-col items-center">
                    <svg width="80" height="80" viewBox="0 0 80 80">
                        <circle cx="40" cy="40" r={r} fill="none" stroke="#e5e7eb" strokeWidth="8" />
                        <circle cx="40" cy="40" r={r} fill="none" stroke={color} strokeWidth="8"
                            strokeDasharray={c} strokeDashoffset={c - (c * value) / 100}
                            strokeLinecap="round" transform="rotate(-90 40 40)" />
                        <text x="40" y="44" textAnchor="middle" fontSize="14" fontWeight="bold" fill={color}>{value}%</text>
                    </svg>
                    <span className="text-xs mt-1 dark:text-gray-300">{label}</span>
                </div>
            ))}
        </div>
    )
}

export default function Dashboard() {
    const [data, setData] = useState<DashboardData | null>(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const fetchData = async () => {
            try {
                const token = localStorage.getItem('token')
                const response = await fetch('/api/visualization/dashboard', {
                    headers: { Authorization: `Bearer ${token}` },
                })
                const result = await response.json()
                setData(result)
            } catch (err) {
                console.error('Failed to fetch dashboard data:', err)
            } finally {
                setLoading(false)
            }
        }
        fetchData()
    }, [])

    if (loading) {
        return (
            <div className="space-y-8" data-testid="dashboard-skeleton">
                <div className="h-64 bg-gray-200 dark:bg-slate-700 rounded-xl animate-pulse" />
                <div className="grid grid-cols-2 gap-8">
                    <div className="h-32 bg-gray-200 dark:bg-slate-700 rounded-xl animate-pulse" />
                    <div className="h-32 bg-gray-200 dark:bg-slate-700 rounded-xl animate-pulse" />
                </div>
            </div>
        )
    }

    if (data === null) {
        return (
            <div className="flex flex-col items-center justify-center h-96 bg-white dark:bg-slate-800 rounded-xl p-6 shadow">
                <p className="text-xl font-semibold text-gray-700 dark:text-gray-300 mb-4">
                    Nie udało się załadować danych.
                </p>
                <p className="text-gray-500 dark:text-gray-400">
                    Spróbuj odświeżyć stronę lub skontaktuj się z pomocą techniczną.
                </p>
            </div>
        )
    }

    const healthScore = data?.healthScore ?? 0
    const anomalies = data?.anomalies ?? []
    const weightData = data?.charts?.weight ?? []
    const rings = data?.charts?.activityRings ?? { move: 0, sleep: 0, diet: 0 }

    return (
        <div className="space-y-8">
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold dark:text-white">Dashboard</h1>
                <NotificationBell />
            </div>

            {/* Health Score */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow">
                <h2 className="text-2xl font-bold mb-4 dark:text-white">Health Score</h2>
                <div className="flex items-center gap-6">
                    <svg className="w-32 h-32" viewBox="0 0 100 100">
                        <circle className="text-gray-200 dark:text-slate-700" strokeWidth="10" stroke="currentColor" fill="transparent" r="40" cx="50" cy="50" />
                        <circle className="text-indigo-600" strokeWidth="10"
                            strokeDasharray={251.2} strokeDashoffset={251.2 - (251.2 * healthScore) / 100}
                            strokeLinecap="round" stroke="currentColor" fill="transparent" r="40" cx="50" cy="50" transform="rotate(-90 50 50)" />
                        <text x="50" y="55" textAnchor="middle" fontSize="22" fontWeight="bold" fill="#6366f1">{healthScore}</text>
                    </svg>
                    <p className="text-lg dark:text-gray-300">Health Score: {healthScore}</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Weight Chart */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow" data-testid="weight-chart">
                    <h2 className="text-xl font-bold mb-4 dark:text-white">Trend wagi</h2>
                    <div className="h-48">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={weightData}>
                                <XAxis dataKey="date" stroke="#6b7280" />
                                <YAxis stroke="#6b7280" />
                                <Tooltip />
                                <Line type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={2} dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Activity Rings */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow">
                    <h2 className="text-xl font-bold mb-4 dark:text-white">Aktywność</h2>
                    <ActivityRings
                        move={rings.move}
                        sleep={rings.sleep}
                        diet={rings.diet}
                    />
                </div>
            </div>

            {/* Anomaly Alerts */}
            {anomalies.length > 0 && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow">
                    <h2 className="text-xl font-bold mb-4 dark:text-white">Anomalie</h2>
                    <ul className="space-y-2">
                        {anomalies.map((anomaly, idx) => (
                            <li key={idx} className="text-red-500 dark:text-red-400 flex items-center">
                                <span className="mr-2">⚠️</span> {anomaly}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    )
}
