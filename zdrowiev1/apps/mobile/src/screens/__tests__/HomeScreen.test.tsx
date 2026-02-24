import React from 'react';
import { render } from '@testing-library/react-native';
import { HomeScreen } from '../HomeScreen';
import { useAuth } from '../../hooks/useAuth';
import { useHealthData } from '../../hooks/useHealthData';
import { useDietData } from '../../hooks/useDietData';

jest.mock('../../hooks/useAuth');
jest.mock('../../hooks/useHealthData');
jest.mock('../../hooks/useDietData');

const mockUseAuth = useAuth as jest.Mock;
const mockUseHealthData = useHealthData as jest.Mock;
const mockUseDietData = useDietData as jest.Mock;

describe('HomeScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockUseAuth.mockReturnValue({
      user: { email: 'jan@test.pl' },
    });

    mockUseHealthData.mockReturnValue({
      healthScore: 85,
      fetchHealthScore: jest.fn(),
      isLoading: false,
    });

    mockUseDietData.mockReturnValue({
      dailySummary: { total: { calories: 1500 } },
      fetchDailySummary: jest.fn(),
      isLoading: false,
    });
  });

  it('H1.1: should greet the user by email', () => {
    const { getByText } = render(<HomeScreen />);
    expect(getByText('Cześć, jan@test.pl')).toBeTruthy();
  });

  it('H1.2: should display health score from useHealthData', () => {
    const { getByText } = render(<HomeScreen />);
    expect(getByText('Health Score: 85')).toBeTruthy();
  });

  it('H1.3: should display calories consumed from useDietData', () => {
    const { getByText } = render(<HomeScreen />);
    expect(getByText('Spożyte kalorie: 1500 kcal')).toBeTruthy();
  });

  it('H1.4: should show loading indicator when data is loading', () => {
    mockUseHealthData.mockReturnValue({ isLoading: true, fetchHealthScore: jest.fn() });
    const { getByTestId } = render(<HomeScreen />);
    expect(getByTestId('home-loading')).toBeTruthy();
  });
});
