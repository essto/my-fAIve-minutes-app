import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { ProfileScreen } from '../ProfileScreen';
import { useAuth } from '../../hooks/useAuth';
import { useProfileStore } from '../../store/useProfileStore';

jest.mock('../../hooks/useAuth');

const mockNavigate = jest.fn();
jest.mock('@react-navigation/native', () => ({
  useNavigation: () => ({
    navigate: mockNavigate,
  }),
}));

describe('ProfileScreen', () => {
  const mockLogout = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    (useAuth as jest.Mock).mockReturnValue({
      user: { email: 'test@example.com', name: 'Jan Kowalski' },
      logout: mockLogout,
    });

    // Reset Zustand store
    const initialState = useProfileStore.getState();
    useProfileStore.setState({ ...initialState, language: 'pl', theme: 'dark' }, true);
  });

  it('P2.1: should render user info and header', () => {
    const { getByText } = render(<ProfileScreen />);
    expect(getByText('Mój Profil')).toBeTruthy();
    expect(getByText('test@example.com')).toBeTruthy();
  });

  it('P2.2: should toggle language', () => {
    const { getByText } = render(<ProfileScreen />);

    expect(useProfileStore.getState().language).toBe('pl');

    fireEvent.press(getByText('Zmień język (PL/EN)'));

    expect(useProfileStore.getState().language).toBe('en');
  });

  it('P2.3: should toggle theme', () => {
    const { getByText } = render(<ProfileScreen />);

    expect(useProfileStore.getState().theme).toBe('dark');

    fireEvent.press(getByText('Zmień motyw'));

    expect(useProfileStore.getState().theme).toBe('light');
  });

  it('P2.4: should call logout on button press', () => {
    const { getByText } = render(<ProfileScreen />);
    fireEvent.press(getByText('Wyloguj się'));
    expect(mockLogout).toHaveBeenCalled();
  });
});
