/* apps/web/src/app/(dashboard)/diagnosis/page.tsx */
'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shared/ui/Card/Card';
import { Button } from '@/components/shared/ui/Button/Button';
import styles from '@/styles/Feature.module.css';

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
        <div className={styles.container}>
            <header className={styles.header}>
                <h1 className={styles.title}>Symptom Checker</h1>
                <p className={styles.subtitle}>Inteligentna analiza Twoich objawów i rekomendacje (AI-powered)</p>
            </header>

            <AnimatePresence mode="wait">
                {step === 1 && (
                    <motion.div
                        key="step1"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                    >
                        <Card glass gradientAccent>
                            <CardContent className="pt-8">
                                <h2 className="text-xl font-bold mb-2">Co Ci dzisiaj dolega?</h2>
                                <p className="text-sm text-color-gray-500 mb-6">Wybierz objawy, które u siebie obserwujesz.</p>
                                <div className="flex flex-wrap gap-3">
                                    {COMMON_SYMPTOMS.map(s => {
                                        const isSelected = !!selectedSymptoms.find(sym => sym.name === s);
                                        return (
                                            <Button
                                                key={s}
                                                variant={isSelected ? 'primary' : 'secondary'}
                                                onClick={() => toggleSymptom(s)}
                                            >
                                                {s}
                                            </Button>
                                        );
                                    })}
                                </div>
                                <div className="mt-10">
                                    <Button
                                        disabled={selectedSymptoms.length === 0}
                                        onClick={() => setStep(2)}
                                        className="w-full"
                                    >
                                        Kontynuuj
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    </motion.div>
                )}

                {step === 2 && (
                    <motion.div
                        key="step2"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                    >
                        <Card glass gradientAccent>
                            <CardHeader>
                                <CardTitle>Szczegóły wybranych objawów</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-6 mb-10">
                                    {selectedSymptoms.map(s => (
                                        <Card key={s.name} interactive className="bg-background/50">
                                            <CardContent className="pt-6">
                                                <h3 className="font-bold mb-4">{s.name}</h3>
                                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                                    <div>
                                                        <label className={styles.label}>Nasilenie (1-10)</label>
                                                        <input
                                                            type="range" min="1" max="10"
                                                            value={s.severity}
                                                            onChange={e => updateSymptom(s.name, { severity: parseInt(e.target.value) })}
                                                            className="w-full mt-2 accent-primary"
                                                        />
                                                        <div className="flex justify-between text-[10px] text-color-gray-500 mt-1">
                                                            <span>Lekkie</span>
                                                            <span>Bardzo silne</span>
                                                        </div>
                                                    </div>
                                                    <div>
                                                        <label htmlFor={`duration-${s.name}`} className={styles.label}>Czas trwania (godz)</label>
                                                        <input
                                                            id={`duration-${s.name}`}
                                                            type="number"
                                                            value={s.durationHours}
                                                            onChange={e => updateSymptom(s.name, { durationHours: parseInt(e.target.value) })}
                                                            className={styles.input}
                                                        />
                                                    </div>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                                <div className="flex gap-4">
                                    <Button variant="outline" onClick={() => setStep(1)} className="flex-1">
                                        Wstecz
                                    </Button>
                                    <Button
                                        isLoading={loading}
                                        onClick={handleReport}
                                        className="flex-[2]"
                                    >
                                        Odbierz Diagnozę
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    </motion.div>
                )}

                {step === 3 && result && (
                    <motion.div
                        key="step3"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                    >
                        <Card
                            glass
                            className="border-t-8"
                            style={{ borderTopColor: result.riskLevel === 'HIGH' ? 'var(--destructive)' : result.riskLevel === 'MEDIUM' ? 'var(--warning)' : 'var(--success)' }}
                        >
                            <CardContent className="pt-8">
                                <div className="text-center mb-8">
                                    <div className={`inline-block px-4 py-1 rounded-full text-xs font-black uppercase tracking-widest mb-4 ${result.riskLevel === 'HIGH' ? 'bg-red-100 text-red-600' :
                                        result.riskLevel === 'MEDIUM' ? 'bg-yellow-100 text-yellow-600' :
                                            'bg-green-100 text-green-600'
                                        }`}>
                                        POZIOM RYZYKA: {result.riskLevel}
                                    </div>
                                    <CardTitle className="text-2xl">Zalecenia medyczne</CardTitle>
                                </div>
                                <div className="p-8 bg-color-gray-100 dark:bg-color-gray-800 rounded-2xl text-lg leading-relaxed italic mb-8 border border-border">
                                    "{result.recommendation}"
                                </div>
                                <div className="p-4 rounded-xl bg-accent/10 border border-accent/20 text-accent text-sm mb-10">
                                    <strong>UWAGA:</strong> To narzędzie służy jedynie do celów informacyjnych i nie zastępuje profesjonalnej porady lekarskiej.
                                </div>
                                <Button
                                    variant="outline"
                                    onClick={() => { setStep(1); setSelectedSymptoms([]); setResult(null); }}
                                    className="w-full"
                                >
                                    Uruchom nowe sprawdzenie
                                </Button>
                            </CardContent>
                        </Card>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
