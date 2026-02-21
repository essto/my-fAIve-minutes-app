'use client'
import { useRouter } from 'next/navigation'
import { useState } from 'react'
import { z } from 'zod'

const schema = z.object({
    email: z.string({ required_error: 'Email jest wymagany' }).min(1, 'Email jest wymagany').email('Nieprawidłowy email'),
    password: z.string({ required_error: 'Hasło jest wymagane' }).min(1, 'Hasło jest wymagane').min(8, 'Hasło musi mieć przynajmniej 8 znaków'),
})

export default function Login() {
    const [form, setForm] = useState({ email: '', password: '' })
    const [errors, setErrors] = useState<Record<string, string>>({})
    const [isLoading, setIsLoading] = useState(false)
    const router = useRouter()

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setIsLoading(true)
        setErrors({})

        try {
            schema.parse(form)
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form),
            })

            if (!response.ok) throw new Error('Login failed')

            const { token } = await response.json()
            localStorage.setItem('token', token)
            router.push('/dashboard')
        } catch (err) {
            if (err instanceof z.ZodError) {
                const newErrors: Record<string, string> = {}
                err.errors.forEach((e) => {
                    if (e.path) newErrors[e.path[0]] = e.message
                })
                setErrors(newErrors)
            } else {
                setErrors({ general: 'Błąd logowania. Sprawdź dane.' })
            }
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center dark:bg-slate-900 bg-gray-50">
            <form
                onSubmit={handleSubmit}
                className="bg-white dark:bg-slate-800 p-8 rounded-xl shadow-lg w-96"
            >
                <h2 className="text-2xl font-bold mb-6 text-center dark:text-white">Logowanie</h2>

                {errors.general && (
                    <p className="text-red-500 mb-4 text-center">{errors.general}</p>
                )}

                <div className="mb-4">
                    <label htmlFor="email" className="block text-sm font-medium dark:text-gray-300 mb-1">Email</label>
                    <input
                        id="email"
                        type="text"
                        autoComplete="email"
                        value={form.email}
                        onChange={(e) => setForm({ ...form, email: e.target.value })}
                        className="w-full p-2 border border-gray-300 rounded-lg dark:bg-slate-700 dark:text-white"
                    />
                    {errors.email && <p className="text-red-500 text-sm mt-1">{errors.email}</p>}
                </div>

                <div className="mb-6">
                    <label htmlFor="password" className="block text-sm font-medium dark:text-gray-300 mb-1">Hasło</label>
                    <input
                        id="password"
                        type="password"
                        value={form.password}
                        onChange={(e) => setForm({ ...form, password: e.target.value })}
                        className="w-full p-2 border border-gray-300 rounded-lg dark:bg-slate-700 dark:text-white"
                    />
                    {errors.password && <p className="text-red-500 text-sm mt-1">{errors.password}</p>}
                </div>

                <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50"
                >
                    {isLoading ? 'Logowanie...' : 'Zaloguj'}
                </button>
            </form>
        </div>
    )
}
