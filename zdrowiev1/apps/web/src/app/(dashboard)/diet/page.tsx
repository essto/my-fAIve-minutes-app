'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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

    return (
        <div className={styles.container}>
            <header className={`${styles.header} flex justify-between items-center`}>
                <div>
                    <h1 className={styles.title}>Twoja Dieta</h1>
                    <p className={styles.subtitle}>Śledź posiłki i osiągaj swoje cele (kcal/makro)</p>
                </div>
                <button
                    onClick={() => setIsAdding(true)}
                    className={`${styles.button} ${styles.primaryButton}`}
                >
                    Zaloguj Posiłek
                </button>
            </header>

            {summary && (
                <section className={styles.grid}>
                    <div className={`${styles.card} ${styles.glass}`}>
                        <p className={styles.label}>Kalorie (dzisiaj)</p>
                        <p className="text-2xl font-black">{summary.total.calories} <span className="text-sm font-normal text-color-gray-500">/ {summary.target.calories} kcal</span></p>
                        <div className="mt-4 h-3 bg-color-gray-100 dark:bg-color-gray-800 rounded-full overflow-hidden border border-border">
                            <div
                                className="h-full bg-primary transition-all duration-1000 ease-out"
                                style={{ width: `${Math.min((summary.total.calories / summary.target.calories) * 100, 100)}%` }}
                            />
                        </div>
                    </div>

                    <div className={styles.card}>
                        <p className={styles.label}>Białko</p>
                        <p className="text-2xl font-bold">{summary.total.protein}g</p>
                        <p className="text-xs text-color-gray-500 mt-1">Cel: {summary.target.protein}g</p>
                    </div>

                    <div className={styles.card}>
                        <p className={styles.label}>Węglowodany</p>
                        <p className="text-2xl font-bold">{summary.total.carbs}g</p>
                        <p className="text-xs text-color-gray-500 mt-1">Cel: {summary.target.carbs}g</p>
                    </div>

                    <div className={styles.card}>
                        <p className={styles.label}>Tłuszcze</p>
                        <p className="text-2xl font-bold">{summary.total.fat}g</p>
                        <p className="text-xs text-color-gray-500 mt-1">Cel: {summary.target.fat}g</p>
                    </div>
                </section>
            )}

            {/* Modal for adding meal */}
            <AnimatePresence>
                {isAdding && (
                    <div className={styles.modalOverlay}>
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 10 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 10 }}
                            className={styles.modal}
                        >
                            <h2 className={styles.sectionTitle}>Zaloguj nowy posiłek</h2>
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
                                    <button
                                        type="button"
                                        onClick={() => setIsAdding(false)}
                                        className={`${styles.button} ${styles.secondaryButton} flex-1`}
                                    >
                                        Anuluj
                                    </button>
                                    <button
                                        type="submit"
                                        className={`${styles.button} ${styles.primaryButton} flex-1`}
                                    >
                                        Zapisz
                                    </button>
                                </div>
                            </form>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
}
