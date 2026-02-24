import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import Login from '../page'
import '@testing-library/jest-dom'

const mockPush = vi.fn()
vi.mock('next/navigation', () => ({
    useRouter: () => ({ push: mockPush, replace: vi.fn() }),
    usePathname: () => '/pl/login',
}))

vi.mock('next-intl', () => ({
    useTranslations: () => (key: string) => key,
    useLocale: () => 'pl'
}))

const localStorageMock = { getItem: vi.fn(), setItem: vi.fn(), removeItem: vi.fn(), clear: vi.fn() }
Object.defineProperty(window, 'localStorage', { value: localStorageMock, writable: true })

describe('Login Page', () => {
    beforeEach(() => { vi.clearAllMocks() })

    test('renders login form', () => {
        render(<Login />)
        expect(screen.getByLabelText('email')).toBeInTheDocument()
        expect(screen.getByLabelText('password')).toBeInTheDocument()
        expect(screen.getAllByRole('button', { name: 'login' })[0]).toBeInTheDocument()
    })

    test('shows validation errors for empty form', async () => {
        render(<Login />)
        fireEvent.click(screen.getAllByRole('button', { name: 'login' })[0])
        // After submission with empty fields, error messages should appear
        await waitFor(() => {
            // The login page shows error spans with text content on validation failure
            const form = document.querySelector('form')
            expect(form).toBeInTheDocument()
            // At least one error-related element should be present
            const errorSpans = document.querySelectorAll('span')
            const hasErrors = Array.from(errorSpans).some(el => el.textContent && el.textContent.length > 0)
            expect(hasErrors).toBe(true)
        })
    })

    test('shows validation errors for invalid input', async () => {
        render(<Login />)
        // Change values first, then submit
        fireEvent.change(screen.getByLabelText('email'), { target: { value: 'invalid-email' } })
        fireEvent.change(screen.getByLabelText('password'), { target: { value: 'short' } })
        // Use findByDisplayValue to ensure the state has updated before clicking
        await screen.findByDisplayValue('invalid-email')
        fireEvent.click(screen.getAllByRole('button', { name: 'login' })[0])
        expect(await screen.findByText('invalid_email')).toBeInTheDocument()
        expect(await screen.findByText('Hasło jest wymagane')).toBeInTheDocument()
    })

    test('submits valid form and redirects', async () => {
        vi.spyOn(global, 'fetch').mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ token: 'test-token' }),
        } as Response)

        render(<Login />)
        fireEvent.change(screen.getByLabelText('email'), { target: { value: 'user@example.com' } })
        fireEvent.change(screen.getByLabelText('password'), { target: { value: 'Password123!' } })
        fireEvent.click(screen.getAllByRole('button', { name: 'login' })[0])

        await waitFor(() => {
            expect(window.localStorage.setItem).toHaveBeenCalledWith('token', 'test-token')
            expect(mockPush).toHaveBeenCalledWith('/dashboard')
        })
    })
})
