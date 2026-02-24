import { useState } from 'react';
import { useTranslations } from 'next-intl';
import { ProfileData, ProfileSchema } from '../useOnboarding';
import { z } from 'zod';
import { motion } from 'framer-motion';

interface Props {
    defaultValues: Partial<ProfileData>;
    onNext: (data: ProfileData) => void;
}

export const ProfileStep = ({ defaultValues, onNext }: Props) => {
    const t = useTranslations('Onboarding');
    const [form, setForm] = useState({
        name: defaultValues.name || '',
        age: defaultValues.age?.toString() || '',
        height: defaultValues.height?.toString() || '',
        weight: defaultValues.weight?.toString() || '',
    });
    const [errors, setErrors] = useState<Record<string, string>>({});

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        setErrors({});

        try {
            const data = {
                name: form.name,
                age: Number(form.age),
                height: Number(form.height),
                weight: Number(form.weight),
            };
            const valid = ProfileSchema.parse(data);
            onNext(valid);
        } catch (error) {
            if (error instanceof z.ZodError) {
                const newErrors: Record<string, string> = {};
                error.errors.forEach((err) => {
                    if (err.path && typeof err.path[0] === 'string') {
                        newErrors[err.path[0]] = err.message;
                    }
                });
                setErrors(newErrors);
            }
        }
    };

    return (
        <motion.form
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            onSubmit={handleSubmit}
            className="space-y-6"
        >
            <div className="text-center mb-6">
                <h2 className="text-2xl font-bold text-foreground">{t('step_1_title')}</h2>
                <p className="text-muted-foreground mt-2">{t('step_1_desc')}</p>
            </div>

            <div className="space-y-4">
                <div className="space-y-1.5">
                    <label htmlFor="name" className="text-sm font-medium text-foreground ml-1">{t('name')}</label>
                    <input
                        id="name"
                        aria-label="name"
                        value={form.name}
                        onChange={(e) => setForm({ ...form, name: e.target.value })}
                        className="w-full bg-neutral-bg3 border border-border rounded-xl px-4 py-3 text-foreground focus:ring-2 focus:ring-brand/50 outline-none transition-all"
                    />
                    {errors.name && <span className="text-xs text-destructive ml-1">{t(errors.name)}</span>}
                </div>

                <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                        <label htmlFor="age" className="text-sm font-medium text-foreground ml-1">{t('age')}</label>
                        <input
                            id="age"
                            type="number"
                            aria-label="age"
                            value={form.age}
                            onChange={(e) => setForm({ ...form, age: e.target.value })}
                            className="w-full bg-neutral-bg3 border border-border rounded-xl px-4 py-3 text-foreground focus:ring-2 focus:ring-brand/50 outline-none transition-all"
                        />
                        {errors.age && <span className="text-xs text-destructive ml-1">{t(errors.age)}</span>}
                    </div>

                    <div className="space-y-1.5">
                        <label htmlFor="height" className="text-sm font-medium text-foreground ml-1">{t('height')} (cm)</label>
                        <input
                            id="height"
                            type="number"
                            aria-label="height"
                            value={form.height}
                            onChange={(e) => setForm({ ...form, height: e.target.value })}
                            className="w-full bg-neutral-bg3 border border-border rounded-xl px-4 py-3 text-foreground focus:ring-2 focus:ring-brand/50 outline-none transition-all"
                        />
                        {errors.height && <span className="text-xs text-destructive ml-1">{t(errors.height)}</span>}
                    </div>
                </div>

                <div className="space-y-1.5">
                    <label htmlFor="weight" className="text-sm font-medium text-foreground ml-1">{t('weight')} (kg)</label>
                    <input
                        id="weight"
                        type="number"
                        aria-label="weight"
                        value={form.weight}
                        onChange={(e) => setForm({ ...form, weight: e.target.value })}
                        className="w-full bg-neutral-bg3 border border-border rounded-xl px-4 py-3 text-foreground focus:ring-2 focus:ring-brand/50 outline-none transition-all"
                    />
                    {errors.weight && <span className="text-xs text-destructive ml-1">{t(errors.weight)}</span>}
                </div>
            </div>

            <button
                type="submit"
                aria-label="next"
                className="w-full mt-6 bg-brand hover:bg-brand-hover text-white font-semibold py-3.5 rounded-xl transition-all shadow-glow hover:-translate-y-0.5 active:translate-y-0"
            >
                {t('next')}
            </button>
        </motion.form>
    );
};
