'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Symptom {
    name: string;
    severity: number;
    durationHours: number;
}

interface TriageResult {
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
    recommendation: string;
}

const COMMON_SYMPTOMS = [
    'Ból głowy', 'Gorączka', 'Kaszel', 'Katar', 'Ból gardła',
    'Nudności', 'Ból brzucha', 'Zmęczenie', 'Ból mięśni', 'Duszność'
];

export default function DiagnosisPage() {
    const [step, setStep] = useState(1);
    const [selectedSymptoms, setSelectedSymptoms] = useState<Symptom[]>([]);
    const [result, setResult] = useState<TriageResult | null>(null);
    const [loading, setLoading] = useState(false);

    const toggleSymptom = (name: string) => {
        if (selectedSymptoms.find(s => s.name === name)) {
            setSelectedSymptoms(selectedSymptoms.filter(s => s.name !== name));
        } else {
            setSelectedSymptoms([...selectedSymptoms, { name, severity: 5, durationHours: 12 }]);
        }
    };

    const updateSymptom = (name: string, fields: Partial<Symptom>) => {
        setSelectedSymptoms(selectedSymptoms.map(s => s.name === name ? { ...s, ...fields } : s));
    };

    const handleReport = async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/diagnosis/report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symptoms: selectedSymptoms }),
            });
            if (res.ok) {
                const data = await res.json();
                setResult(data.triage);
                setStep(3);
            }
        } catch (err) {
            console.error('Failed to report symptoms:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-neutral-50 dark:bg-neutral-950 p-6">
            <div className="max-w-2xl mx-auto">
                <header className="mb-12 text-center">
                    <h1 className="text-4xl font-extrabold text-neutral-900 dark:text-white tracking-tight">Symptom Checker</h1>
                    <p className="text-neutral-500 mt-2">Inteligentna analiza Twoich objawów</p>
                </header>

                <AnimatePresence mode="wait">
                    {step === 1 && (
                        <motion.div
                            key="step1"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            className="bg-white dark:bg-neutral-900 p-8 rounded-3xl shadow-xl border border-neutral-200 dark:border-neutral-800"
                        >
                            <h2 className="text-2xl font-bold mb-6">Co Ci dolega?</h2>
                            <div className="grid grid-cols-2 gap-3 mb-8">
                                {COMMON_SYMPTOMS.map(s => (
                                    <button
                                        key={s}
                                        onClick={() => toggleSymptom(s)}
                                        className={`p-4 rounded-2xl text-left transition-all ${selectedSymptoms.find(sym => sym.name === s)
                                            ? 'bg-indigo-600 text-white shadow-lg'
                                            : 'bg-neutral-100 dark:bg-neutral-800 hover:bg-neutral-200 dark:hover:bg-neutral-700'
                                            }`}
                                    >
                                        {s}
                                    </button>
                                ))}
                            </div>
                            <button
                                disabled={selectedSymptoms.length === 0}
                                onClick={() => setStep(2)}
                                className="w-full py-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-2xl font-bold disabled:opacity-50 transition-all"
                            >
                                Kontynuuj
                            </button>
                        </motion.div>
                    )}

                    {step === 2 && (
                        <motion.div
                            key="step2"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            className="bg-white dark:bg-neutral-900 p-8 rounded-3xl shadow-xl border border-neutral-200 dark:border-neutral-800"
                        >
                            <h2 className="text-2xl font-bold mb-6">Szczegóły objawów</h2>
                            <div className="space-y-6 mb-8">
                                {selectedSymptoms.map(s => (
                                    <div key={s.name} className="p-4 bg-neutral-50 dark:bg-neutral-800/50 rounded-2xl">
                                        <h3 className="font-bold mb-3">{s.name}</h3>
                                        <div className="grid grid-cols-2 gap-4">
                                            <div>
                                                <label className="text-xs text-neutral-500 uppercase font-bold">Nasilenie (1-10)</label>
                                                <input
                                                    type="range" min="1" max="10"
                                                    value={s.severity}
                                                    onChange={e => updateSymptom(s.name, { severity: parseInt(e.target.value) })}
                                                    className="w-full mt-1"
                                                />
                                            </div>
                                            <div>
                                                <label className="text-xs text-neutral-500 uppercase font-bold">Czas trwania (godz)</label>
                                                <input
                                                    type="number"
                                                    value={s.durationHours}
                                                    onChange={e => updateSymptom(s.name, { durationHours: parseInt(e.target.value) })}
                                                    className="w-full bg-transparent border-b border-neutral-300 dark:border-neutral-700 p-1"
                                                />
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <div className="flex gap-4">
                                <button onClick={() => setStep(1)} className="flex-1 py-4 text-neutral-500 font-bold">Wstecz</button>
                                <button
                                    onClick={handleReport}
                                    disabled={loading}
                                    className="flex-[2] py-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-2xl font-bold shadow-lg"
                                >
                                    {loading ? 'Analizowanie...' : 'Odbierz Diagnozę'}
                                </button>
                            </div>
                        </motion.div>
                    )}

                    {step === 3 && result && (
                        <motion.div
                            key="step3"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="bg-white dark:bg-neutral-900 p-10 rounded-3xl shadow-2xl border-t-8 border-indigo-600"
                        >
                            <div className="text-center mb-8">
                                <div className={`inline-block px-6 py-2 rounded-full text-sm font-black uppercase tracking-widest mb-4 ${result.riskLevel === 'HIGH' ? 'bg-red-100 text-red-600' :
                                    result.riskLevel === 'MEDIUM' ? 'bg-yellow-100 text-yellow-600' :
                                        'bg-green-100 text-green-600'
                                    }`}>
                                    RYZYKO: {result.riskLevel}
                                </div>
                                <h2 className="text-3xl font-bold">Zalecenia</h2>
                            </div>
                            <div className="p-6 bg-neutral-50 dark:bg-neutral-800 rounded-2xl text-lg leading-relaxed text-neutral-700 dark:text-neutral-300 italic mb-8">
                                "{result.recommendation}"
                            </div>
                            <button
                                onClick={() => { setStep(1); setSelectedSymptoms([]); setResult(null); }}
                                className="w-full py-4 text-indigo-600 font-bold border-2 border-indigo-600 rounded-2xl hover:bg-indigo-50 dark:hover:bg-indigo-900/30 transition-all"
                            >
                                Nowe Sprawdzenie
                            </button>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
