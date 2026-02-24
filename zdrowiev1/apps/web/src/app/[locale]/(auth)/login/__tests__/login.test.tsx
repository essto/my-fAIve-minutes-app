import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import Login from '../page'
import '@testing-library/jest-dom'

const mockPush = vi.fn()
vi.mock('next/navigation', () => ({
    useRouter: () => ({ push: mockPush }),
}))

const localStorageMock = { getItem: vi.fn(), setItem: vi.fn(), removeItem: vi.fn(), clear: vi.fn() }
Object.defineProperty(window, 'localStorage', { value: localStorageMock, writable: true })

describe('Login Page', () => {
    beforeEach(() => { vi.clearAllMocks() })

    test('renders login form', () => {
        render(<Login />)
        expect(screen.getByLabelText('Email')).toBeInTheDocument()
        expect(screen.getByLabelText('Hasło')).toBeInTheDocument()
        expect(screen.getByRole('button', { name: 'Zaloguj' })).toBeInTheDocument()
    })

    test('shows validation errors for empty form', async () => {
        render(<Login />)
        fireEvent.click(screen.getByRole('button', { name: 'Zaloguj' }))
        // Find any validation error - errors container should have some error text
        await waitFor(() => {
            const errorEls = document.querySelectorAll('p.text-red-500')
            expect(errorEls.length).toBeGreaterThan(0)
        })
    })

    test('shows validation errors for invalid input', async () => {
        render(<Login />)
        // Change values first, then submit
        fireEvent.change(screen.getByLabelText('Email'), { target: { value: 'invalid-email' } })
        fireEvent.change(screen.getByLabelText('Hasło'), { target: { value: 'short' } })
        // Use findByDisplayValue to ensure the state has updated before clicking
        await screen.findByDisplayValue('invalid-email')
        fireEvent.click(screen.getByRole('button', { name: 'Zaloguj' }))
        expect(await screen.findByText('Nieprawidłowy email')).toBeInTheDocument()
        expect(await screen.findByText('Hasło musi mieć przynajmniej 8 znaków')).toBeInTheDocument()
    })

    test('submits valid form and redirects', async () => {
        vi.spyOn(global, 'fetch').mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ token: 'test-token' }),
        } as Response)

        render(<Login />)
        fireEvent.change(screen.getByLabelText('Email'), { target: { value: 'user@example.com' } })
        fireEvent.change(screen.getByLabelText('Hasło'), { target: { value: 'Password123!' } })
        fireEvent.click(screen.getByRole('button', { name: 'Zaloguj' }))

        await waitFor(() => {
            expect(window.localStorage.setItem).toHaveBeenCalledWith('token', 'test-token')
            expect(mockPush).toHaveBeenCalledWith('/dashboard')
        })
    })
})
