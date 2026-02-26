import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { LoginScreen } from '../LoginScreen';
import { useAuth } from '../../hooks/useAuth';

jest.mock('../../hooks/useAuth');

const mockUseAuth = useAuth as jest.Mock;

describe('LoginScreen', () => {
  const mockLogin = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseAuth.mockReturnValue({
      login: mockLogin,
      isLoading: false,
      error: null,
      isAuthenticated: false,
    });
  });

  it('L1.1: should render email and password inputs and login button', () => {
    const { getByPlaceholderText, getByText } = render(<LoginScreen />);

    expect(getByPlaceholderText('Email')).toBeTruthy();
    expect(getByPlaceholderText('Hasło')).toBeTruthy();
    expect(getByText('Zaloguj się')).toBeTruthy();
  });

  it('L1.2: should call useAuth login when form is submitted', async () => {
    const { getByPlaceholderText, getByText } = render(<LoginScreen />);

    fireEvent.changeText(getByPlaceholderText('Email'), 'test@zdrowie.pl');
    fireEvent.changeText(getByPlaceholderText('Hasło'), 'haslo123');
    fireEvent.press(getByText('Zaloguj się'));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith('test@zdrowie.pl', 'haslo123');
    });
  });

  it('L1.3: should display error message from useAuth', () => {
    mockUseAuth.mockReturnValue({
      login: mockLogin,
      isLoading: false,
      error: 'Nieprawidłowe dane logowania',
      isAuthenticated: false,
    });

    const { getByTestId } = render(<LoginScreen />);
    expect(getByTestId('login-error')).toBeTruthy();
  });

  it('L1.4: should show loading indicator when isLoading is true', () => {
    mockUseAuth.mockReturnValue({
      login: mockLogin,
      isLoading: true,
      error: null,
      isAuthenticated: false,
    });

    const { getByTestId, queryByText } = render(<LoginScreen />);
    expect(getByTestId('loading-indicator')).toBeTruthy();
    expect(queryByText('Zaloguj się')).toBeNull(); // Button text hides while loading
  });
});
