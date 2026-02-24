'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { z } from 'zod';
import Link from 'next/link';
import { motion } from 'framer-motion';

const schema = z.object({
    email: z.string({ required_error: 'Email jest wymagany' }).min(1, 'Email jest wymagany').email('Nieprawidłowy email'),
    password: z.string({ required_error: 'Hasło jest wymagane' }).min(1, 'Hasło jest wymagane').min(8, 'Hasło musi mieć przynajmniej 8 znaków'),
});

export default function Login() {
    const [form, setForm] = useState({ email: '', password: '' });
    const [errors, setErrors] = useState<Record<string, string>>({});
    const [isLoading, setIsLoading] = useState(false);
    const router = useRouter();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setErrors({});

        try {
            schema.parse(form);
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form),
            });

            if (!response.ok) throw new Error('Login failed');

            const { token } = await response.json();
            localStorage.setItem('token', token);
            router.push('/dashboard');
        } catch (err) {
            if (err instanceof z.ZodError) {
                const newErrors: Record<string, string> = {};
                err.errors.forEach((e) => {
                    if (e.path) newErrors[e.path[0] as string] = e.message;
                });
                setErrors(newErrors);
            } else {
                setErrors({ general: 'Błąd logowania. Sprawdź dane.' });
            }
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center p-4 bg-background w-full">
            <motion.div
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
                className="w-full max-w-md glass-card p-8 md:p-10 relative overflow-hidden"
            >
                {/* Decorative Premium Top Border */}
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-brand to-brand-light"></div>

                <div className="mb-8 text-center mt-2">
                    <h1 className="text-2xl md:text-3xl font-bold text-foreground tracking-tight mb-2">Witaj ponownie</h1>
                    <p className="text-muted-foreground text-sm">Zaloguj się, aby kontynuować</p>
                </div>

                <form onSubmit={handleSubmit} noValidate className="space-y-5">
                    {errors.general && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm text-center font-medium"
                        >
                            {errors.general}
                        </motion.div>
                    )}

                    <div className="space-y-1.5">
                        <label htmlFor="email" className="text-sm font-medium text-foreground ml-1">Email</label>
                        <input
                            id="email"
                            type="email"
                            placeholder="Twój adres email"
                            autoComplete="email"
                            value={form.email}
                            onChange={(e) => setForm({ ...form, email: e.target.value })}
                            className="w-full bg-neutral-bg3 border border-border rounded-xl px-4 py-3.5 text-foreground placeholder:text-muted-foreground focus:ring-2 focus:ring-brand/50 focus:border-brand outline-none transition-all disabled:opacity-50"
                            disabled={isLoading}
                        />
                        {errors.email && <span className="text-xs text-destructive ml-1">{errors.email}</span>}
                    </div>

                    <div className="space-y-1.5">
                        <label htmlFor="password" className="text-sm font-medium text-foreground ml-1">Hasło</label>
                        <input
                            id="password"
                            type="password"
                            placeholder="••••••••"
                            autoComplete="current-password"
                            value={form.password}
                            onChange={(e) => setForm({ ...form, password: e.target.value })}
                            className="w-full bg-neutral-bg3 border border-border rounded-xl px-4 py-3.5 text-foreground placeholder:text-muted-foreground focus:ring-2 focus:ring-brand/50 focus:border-brand outline-none transition-all disabled:opacity-50"
                            disabled={isLoading}
                        />
                        {errors.password && <span className="text-xs text-destructive ml-1">{errors.password}</span>}
                    </div>

                    <button
                        type="submit"
                        disabled={isLoading}
                        className="w-full mt-4 bg-brand hover:bg-brand-hover text-white font-semibold py-3.5 rounded-xl transition-all shadow-glow focus:ring-2 focus:ring-offset-2 focus:ring-brand focus:ring-offset-background disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0"
                    >
                        {isLoading ? 'Logowanie...' : 'Zaloguj się'}
                    </button>
                </form>

                <div className="mt-8 text-center text-sm text-muted-foreground">
                    Nie masz konta?{' '}
                    <Link href="/register" className="text-brand hover:text-brand-light font-semibold transition-colors">
                        Zarejestruj się
                    </Link>
                </div>
            </motion.div>
        </div>
    );
}
