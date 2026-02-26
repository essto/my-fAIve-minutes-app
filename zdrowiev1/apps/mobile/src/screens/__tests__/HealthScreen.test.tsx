import React from 'react';
import { render } from '@testing-library/react-native';
import { HealthScreen } from '../HealthScreen';
import { useHealthData } from '../../hooks/useHealthData';

jest.mock('../../hooks/useHealthData');
const mockUseHealthData = useHealthData as jest.Mock;

describe('HealthScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseHealthData.mockReturnValue({
      healthScore: 90,
      healthBreakdown: { weight: 95, sleep: 81, activity: 95 },
      weightHistory: [{ id: '1', value: 80, timestamp: '2026-02-01' }],
      anomalies: [],
      isLoading: false,
      fetchHealthScore: jest.fn(),
      fetchWeightHistory: jest.fn(),
      fetchAnomalies: jest.fn(),
    });
  });

  it('HE1.1: should display health score', () => {
    const { getByText } = render(<HealthScreen />);
    expect(getByText(/Health Score:/i)).toBeTruthy();
    expect(getByText('90')).toBeTruthy();
  });

  it('HE1.2: should render weight card and latest weight', () => {
    const { getAllByText, getByText } = render(<HealthScreen />);
    expect(getAllByText(/Waga/i).length).toBeGreaterThan(0);
    expect(getByText('80')).toBeTruthy();
    expect(getByText('kg')).toBeTruthy();
  });

  it('HE1.3: should render anomalies if they exist', () => {
    mockUseHealthData.mockReturnValue({
      ...mockUseHealthData(),
      anomalies: [
        { id: 'a1', metric: 'Tętno', value: 120, severity: 'high', message: 'Wysokie tętno' },
      ],
      fetchHealthScore: jest.fn(),
      fetchWeightHistory: jest.fn(),
      fetchAnomalies: jest.fn(),
    });

    const { getByText } = render(<HealthScreen />);
    expect(getByText(/Wykryto Anomalie/i)).toBeTruthy();
    expect(getByText(/Wysokie tętno/i)).toBeTruthy();
  });

  it('HE1.4: should show loading state', () => {
    mockUseHealthData.mockReturnValue({
      isLoading: true,
      fetchHealthScore: jest.fn(),
      fetchWeightHistory: jest.fn(),
      fetchAnomalies: jest.fn(),
    });

    const { getByTestId } = render(<HealthScreen />);
    expect(getByTestId('health-loading')).toBeTruthy();
  });
});
