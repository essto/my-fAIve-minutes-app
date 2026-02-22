'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { z } from 'zod';
import Link from 'next/link';
import styles from '../../../styles/Auth.module.css';

const schema = z.object({
    email: z.string({ required_error: 'Email jest wymagany' }).min(1, 'Email jest wymagany').email('Nieprawidłowy email'),
    password: z.string({ required_error: 'Hasło jest wymagane' }).min(1, 'Hasło jest wymagane').min(8, 'Hasło musi mieć przynajmniej 8 znaków'),
});

export default function Register() {
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
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Registration failed');
            }

            router.push('/login?registered=true');
        } catch (err: any) {
            if (err instanceof z.ZodError) {
                const newErrors: Record<string, string> = {};
                err.errors.forEach((e) => {
                    if (e.path) newErrors[e.path[0] as string] = e.message;
                });
                setErrors(newErrors);
            } else {
                setErrors({ general: err.message || 'Błąd rejestracji.' });
            }
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className={styles.authContainer}>
            <div className={`${styles.authCard} ${styles.glass}`}>
                <h1 className={styles.title}>Dołącz do nas</h1>
                <p className={styles.subtitle}>Rozpocznij monitorowanie swojego zdrowia</p>

                <form onSubmit={handleSubmit} noValidate>
                    {errors.general && (
                        <div className={styles.generalError} id="general-error">{errors.general}</div>
                    )}

                    <div className={styles.formGroup}>
                        <label htmlFor="email" className={styles.label}>Email</label>
                        <input
                            id="email"
                            type="email"
                            placeholder="Twój adres email"
                            autoComplete="email"
                            value={form.email}
                            onChange={(e) => setForm({ ...form, email: e.target.value })}
                            className={styles.input}
                            disabled={isLoading}
                        />
                        {errors.email && <span className={styles.errorText}>{errors.email}</span>}
                    </div>

                    <div className={styles.formGroup}>
                        <label htmlFor="password" className={styles.label}>Hasło</label>
                        <input
                            id="password"
                            type="password"
                            placeholder="Min. 8 znaków"
                            autoComplete="new-password"
                            value={form.password}
                            onChange={(e) => setForm({ ...form, password: e.target.value })}
                            className={styles.input}
                            disabled={isLoading}
                        />
                        {errors.password && <span className={styles.errorText}>{errors.password}</span>}
                    </div>

                    <button
                        id="register-submit"
                        type="submit"
                        disabled={isLoading}
                        className={styles.button}
                    >
                        {isLoading ? 'Rejestrowanie...' : 'Zarejestruj się'}
                    </button>
                </form>

                <div className={styles.linkContainer}>
                    Masz już konto? <Link href="/login" className={styles.link}>Zaloguj się</Link>
                </div>
            </div>
        </div>
    );
}
