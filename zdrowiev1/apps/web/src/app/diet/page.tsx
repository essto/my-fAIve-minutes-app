'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

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
    const [meals, setMeals] = useState<Meal[]>([]);
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
                        protein: 10, // Mocked for simplicity in UI test
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
        <div className="min-h-screen bg-neutral-50 dark:bg-neutral-950 p-6">
            <div className="max-w-4xl mx-auto space-y-8">
                <header className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold text-neutral-900 dark:text-white">Twoja Dieta</h1>
                        <p className="text-neutral-500">Śledź posiłki i osiągaj swoje cele</p>
                    </div>
                    <button
                        onClick={() => setIsAdding(true)}
                        className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white rounded-full font-medium transition-colors shadow-lg"
                    >
                        Zaloguj Posiłek
                    </button>
                </header>

                {summary && (
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div className="bg-white dark:bg-neutral-900 p-6 rounded-2xl shadow-sm border border-neutral-200 dark:border-neutral-800">
                            <p className="text-sm text-neutral-500 mb-1">Kalorie</p>
                            <p className="text-2xl font-bold">{summary.total.calories} / {summary.target.calories}</p>
                            <div className="mt-2 h-2 bg-neutral-100 dark:bg-neutral-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-green-500 transition-all duration-500"
                                    style={{ width: `${Math.min((summary.total.calories / summary.target.calories) * 100, 100)}%` }}
                                />
                            </div>
                        </div>
                        {/* Macro Stats */}
                        {['Białko', 'Węgle', 'Tłuszcze'].map((label, idx) => {
                            const keys = ['protein', 'carbs', 'fat'] as const;
                            const key = keys[idx];
                            return (
                                <div key={label} className="bg-white dark:bg-neutral-900 p-6 rounded-2xl shadow-sm border border-neutral-200 dark:border-neutral-800">
                                    <p className="text-sm text-neutral-500 mb-1">{label}</p>
                                    <p className="text-2xl font-bold">{summary.total[key]}g</p>
                                </div>
                            );
                        })}
                    </div>
                )}

                {/* Modal for adding meal */}
                <AnimatePresence>
                    {isAdding && (
                        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                className="bg-white dark:bg-neutral-900 w-full max-w-md p-8 rounded-3xl shadow-2xl border border-neutral-200 dark:border-neutral-800"
                            >
                                <h2 className="text-2xl font-bold mb-6">Zaloguj Posiłek</h2>
                                <form onSubmit={handleLogMeal} className="space-y-4">
                                    <div>
                                        <label htmlFor="meal-name" className="block text-sm font-medium mb-1">Nazwa Posiłku</label>
                                        <input
                                            id="meal-name"
                                            type="text"
                                            className="w-full p-3 rounded-xl border dark:bg-neutral-800 dark:border-neutral-700"
                                            placeholder="np. Śniadanie"
                                            value={newMeal.name}
                                            onChange={(e) => setNewMeal({ ...newMeal, name: e.target.value })}
                                            required
                                        />
                                    </div>
                                    <div>
                                        <label htmlFor="product-name" className="block text-sm font-medium mb-1">Produkt</label>
                                        <input
                                            id="product-name"
                                            type="text"
                                            className="w-full p-3 rounded-xl border dark:bg-neutral-800 dark:border-neutral-700"
                                            placeholder="np. Jajecznica"
                                            value={newMeal.productName}
                                            onChange={(e) => setNewMeal({ ...newMeal, productName: e.target.value })}
                                            required
                                        />
                                    </div>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label htmlFor="product-quantity" className="block text-sm font-medium mb-1">Waga (g)</label>
                                            <input
                                                id="product-quantity"
                                                type="number"
                                                className="w-full p-3 rounded-xl border dark:bg-neutral-800 dark:border-neutral-700"
                                                value={newMeal.quantity}
                                                onChange={(e) => setNewMeal({ ...newMeal, quantity: parseInt(e.target.value) })}
                                                required
                                            />
                                        </div>
                                        <div>
                                            <label htmlFor="product-calories" className="block text-sm font-medium mb-1">Kalorie</label>
                                            <input
                                                id="product-calories"
                                                type="number"
                                                className="w-full p-3 rounded-xl border dark:bg-neutral-800 dark:border-neutral-700"
                                                value={newMeal.calories}
                                                onChange={(e) => setNewMeal({ ...newMeal, calories: parseInt(e.target.value) })}
                                                required
                                            />
                                        </div>
                                    </div>
                                    <div className="flex gap-4 pt-4">
                                        <button
                                            type="button"
                                            onClick={() => setIsAdding(false)}
                                            className="flex-1 py-3 text-neutral-500 font-medium"
                                        >
                                            Anuluj
                                        </button>
                                        <button
                                            type="submit"
                                            className="flex-1 py-3 bg-green-600 hover:bg-green-700 text-white rounded-xl font-bold transition-shadow"
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
        </div>
    );
}
