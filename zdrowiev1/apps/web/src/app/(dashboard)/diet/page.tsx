/* apps/web/src/app/(dashboard)/diet/page.tsx */
'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shared/ui/Card/Card';
import { SkeletonLoader } from '@/components/shared/ui/SkeletonLoader/SkeletonLoader';
import { Button } from '@/components/shared/ui/Button/Button';
import styles from '@/styles/Feature.module.css';

interface Product {
    name: string;
    quantity: number;
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
}

interface Meal {
    id: string;
    name: string;
    consumedAt: string;
    products: Product[];
}

interface Summary {
    total: { calories: number; protein: number; carbs: number; fat: number };
    target: { calories: number; protein: number; carbs: number; fat: number };
    isDeficit: boolean;
    isSurplus: boolean;
}

export default function DietPage() {
    const [summary, setSummary] = useState<Summary | null>(null);
    const [isAdding, setIsAdding] = useState(false);
    const [loading, setLoading] = useState(true);
    const [newMeal, setNewMeal] = useState({ name: '', productName: '', quantity: 100, calories: 100 });

    const fetchSummary = async () => {
        try {
            const res = await fetch('/api/diet/summary');
            if (res.ok) {
                const data = await res.json();
                setSummary(data);
            }
        } catch (err) {
            console.error('Failed to fetch summary:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleLogMeal = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            const res = await fetch('/api/diet', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: newMeal.name,
                    products: [{
                        name: newMeal.productName,
                        quantity: newMeal.quantity,
                        calories: newMeal.calories,
                        protein: 10,
                        carbs: 10,
                        fat: 5
                    }]
                }),
            });

            if (res.ok) {
                setIsAdding(false);
                setNewMeal({ name: '', productName: '', quantity: 100, calories: 100 });
                await fetchSummary();
            }
        } catch (err) {
            console.error('Failed to log meal:', err);
        }
    };

    useEffect(() => {
        fetchSummary();
    }, []);

    if (loading) {
        return (
            <div className={styles.container}>
                <header className={`${styles.header} flex justify-between items-center`}>
                    <div>
                        <SkeletonLoader className="h-10 w-48 mb-2" />
                        <SkeletonLoader className="h-4 w-64" />
                    </div>
                </header>
                <div className={styles.grid}>
                    {[1, 2, 3, 4].map(i => (
                        <Card key={i}>
                            <CardContent className="h-32 flex items-center justify-center">
                                <SkeletonLoader className="h-full w-full" />
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div className={styles.container}>
            <header className={`${styles.header} flex justify-between items-center`}>
                <div>
                    <h1 className={styles.title}>Twoja Dieta</h1>
                    <p className={styles.subtitle}>Śledź posiłki i osiągaj swoje cele (kcal/makro)</p>
                </div>
                <Button onClick={() => setIsAdding(true)}>
                    Zaloguj Posiłek
                </Button>
            </header>

            {summary && (
                <section className={styles.grid}>
                    <Card glass gradientAccent>
                        <CardContent className="pt-6">
                            <p className={styles.label}>Kalorie (dzisiaj)</p>
                            <p className="text-2xl font-black">{summary.total.calories} <span className="text-sm font-normal text-color-gray-500">/ {summary.target.calories} kcal</span></p>
                            <div className="mt-4 h-3 bg-color-gray-100 dark:bg-color-gray-800 rounded-full overflow-hidden border border-border">
                                <div
                                    className="h-full bg-primary transition-all duration-1000 ease-out"
                                    style={{ width: `${Math.min((summary.total.calories / summary.target.calories) * 100, 100)}%` }}
                                />
                            </div>
                        </CardContent>
                    </Card>

                    <Card interactive>
                        <CardContent className="pt-6">
                            <p className={styles.label}>Białko</p>
                            <p className="text-2xl font-bold">{summary.total.protein}g</p>
                            <p className="text-xs text-color-gray-500 mt-1">Cel: {summary.target.protein}g</p>
                        </CardContent>
                    </Card>

                    <Card interactive>
                        <CardContent className="pt-6">
                            <p className={styles.label}>Węglowodany</p>
                            <p className="text-2xl font-bold">{summary.total.carbs}g</p>
                            <p className="text-xs text-color-gray-500 mt-1">Cel: {summary.target.carbs}g</p>
                        </CardContent>
                    </Card>

                    <Card interactive>
                        <CardContent className="pt-6">
                            <p className={styles.label}>Tłuszcze</p>
                            <p className="text-2xl font-bold">{summary.total.fat}g</p>
                            <p className="text-xs text-color-gray-500 mt-1">Cel: {summary.target.fat}g</p>
                        </CardContent>
                    </Card>
                </section>
            )}

            {/* Modal for adding meal using Card design */}
            <AnimatePresence>
                {isAdding && (
                    <div className={styles.modalOverlay}>
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 10 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 10 }}
                            className="w-full max-w-[500px]"
                        >
                            <Card glass>
                                <CardHeader>
                                    <CardTitle>Zaloguj nowy posiłek</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <form onSubmit={handleLogMeal} className="space-y-4">
                                        <div>
                                            <label htmlFor="meal-name" className={styles.label}>Nazwa Posiłku</label>
                                            <input
                                                id="meal-name"
                                                type="text"
                                                className={styles.input}
                                                placeholder="np. Śniadanie energetyczne"
                                                value={newMeal.name}
                                                onChange={(e) => setNewMeal({ ...newMeal, name: e.target.value })}
                                                required
                                            />
                                        </div>
                                        <div>
                                            <label htmlFor="product-name" className={styles.label}>Główny produkt</label>
                                            <input
                                                id="product-name"
                                                type="text"
                                                className={styles.input}
                                                placeholder="np. Owsianka z borówkami"
                                                value={newMeal.productName}
                                                onChange={(e) => setNewMeal({ ...newMeal, productName: e.target.value })}
                                                required
                                            />
                                        </div>
                                        <div className="grid grid-cols-2 gap-4">
                                            <div>
                                                <label htmlFor="product-quantity" className={styles.label}>Waga (g)</label>
                                                <input
                                                    id="product-quantity"
                                                    type="number"
                                                    className={styles.input}
                                                    value={newMeal.quantity}
                                                    onChange={(e) => setNewMeal({ ...newMeal, quantity: parseInt(e.target.value) })}
                                                    required
                                                />
                                            </div>
                                            <div>
                                                <label htmlFor="product-calories" className={styles.label}>Kalorie (kcal)</label>
                                                <input
                                                    id="product-calories"
                                                    type="number"
                                                    className={styles.input}
                                                    value={newMeal.calories}
                                                    onChange={(e) => setNewMeal({ ...newMeal, calories: parseInt(e.target.value) })}
                                                    required
                                                />
                                            </div>
                                        </div>
                                        <div className="flex gap-4 pt-6">
                                            <Button
                                                variant="outline"
                                                type="button"
                                                onClick={() => setIsAdding(false)}
                                                className="flex-1"
                                            >
                                                Anuluj
                                            </Button>
                                            <Button
                                                variant="primary"
                                                type="submit"
                                                className="flex-1"
                                            >
                                                Zapisz
                                            </Button>
                                        </div>
                                    </form>
                                </CardContent>
                            </Card>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
}
