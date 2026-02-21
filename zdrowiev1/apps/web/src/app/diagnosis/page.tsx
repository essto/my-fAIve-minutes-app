'use client'
import { useState } from 'react'

const symptoms = ['Ból głowy', 'Gorączka', 'Zawroty głowy', 'Nudności', 'Wysypka', 'Ból brzucha', 'Ból mięśni', 'Dreszcze', 'Katar', 'Kaszel']
type Triage = 'ZIELONY' | 'ŻÓŁTY' | 'CZERWONY'

const badgeColor: Record<Triage, string> = {
    ZIELONY: 'bg-green-500',
    ŻÓŁTY: 'bg-yellow-500',
    CZERWONY: 'bg-red-500',
}

export default function DiagnosisPage() {
    const [selected, setSelected] = useState<string[]>([])
    const [result, setResult] = useState<{ triage: Triage; description: string } | null>(null)
    const [isLoading, setIsLoading] = useState(false)

    const toggle = (s: string) =>
        setSelected(prev => prev.includes(s) ? prev.filter(x => x !== s) : [...prev, s])

    const handleCheck = async () => {
        setIsLoading(true)
        try {
            const response = await fetch('/api/diagnosis/check', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symptoms: selected }),
            })
            setResult(await response.json())
        } catch (err) {
            console.error('Diagnosis error:', err)
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="space-y-8">
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow">
                <h2 className="text-xl font-bold mb-4 dark:text-white">Wybierz objawy</h2>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                    {symptoms.map(s => (
                        <div key={s} className="flex items-center gap-2">
                            <input type="checkbox" id={s} checked={selected.includes(s)} onChange={() => toggle(s)}
                                className="h-5 w-5 text-indigo-600 rounded" aria-label={s} />
                            <label htmlFor={s} className="dark:text-white cursor-pointer">{s}</label>
                        </div>
                    ))}
                </div>
                <button onClick={handleCheck} disabled={isLoading || selected.length === 0}
                    className="mt-6 bg-indigo-600 text-white py-2 px-6 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50">
                    {isLoading ? 'Analizowanie...' : 'Sprawdź'}
                </button>
            </div>

            {result && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow">
                    <h2 className="text-xl font-bold mb-4 dark:text-white">Wynik diagnozy</h2>
                    <div className="flex items-center gap-4 mb-4">
                        <span className={`px-3 py-1 rounded-full text-white font-bold ${badgeColor[result.triage]}`}>{result.triage}</span>
                        <p className="dark:text-white">{result.description}</p>
                    </div>
                    <div>
                        <h3 className="font-medium mb-2 dark:text-white">Zalecane działania:</h3>
                        <ul className="space-y-1 dark:text-gray-300">
                            {result.triage === 'ZIELONY' && <li>• Obserwuj objawy w domu</li>}
                            {result.triage === 'ŻÓŁTY' && (
                                <><li>• Skonsultuj się z lekarzem w ciągu 24h</li><li>• Monitoruj objawy</li></>
                            )}
                            {result.triage === 'CZERWONY' && (
                                <><li>• Natychmiast skontaktuj się z pogotowiem</li><li>• Nie pozostawaj sam</li></>
                            )}
                        </ul>
                    </div>
                </div>
            )}
        </div>
    )
}
