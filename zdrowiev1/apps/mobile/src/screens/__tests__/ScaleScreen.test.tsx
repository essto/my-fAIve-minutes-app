import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { ScaleScreen } from '../ScaleScreen';
import { useBLE } from '../../hooks/useBLE';

jest.mock('../../hooks/useBLE');
const mockUseBLE = useBLE as jest.Mock;

describe('ScaleScreen', () => {
  const mockStartScan = jest.fn();
  const mockConnect = jest.fn();
  const mockDisconnect = jest.fn();
  const mockOnWeightReading = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseBLE.mockReturnValue({
      devices: [],
      connectedDevice: null,
      isScanning: false,
      error: null,
      lastReading: null,
      startScan: mockStartScan,
      connectToDevice: mockConnect,
      disconnect: mockDisconnect,
      onWeightReading: mockOnWeightReading,
    });
  });

  it('SS1.1: should render empty state and scan button initially', () => {
    const { getByText } = render(<ScaleScreen />);
    expect(getByText('Wyszukaj urządzenie')).toBeTruthy();
    expect(getByText('Brak urządzeń w pobliżu')).toBeTruthy();
  });

  it('SS1.2: should call startScan when button pressed', () => {
    const { getByText } = render(<ScaleScreen />);
    fireEvent.press(getByText('Wyszukaj urządzenie'));
    expect(mockStartScan).toHaveBeenCalled();
  });

  it('SS1.3: should show scanning indicator', () => {
    mockUseBLE.mockReturnValue({
      ...mockUseBLE(),
      isScanning: true,
    });
    const { getByTestId, getByText } = render(<ScaleScreen />);
    expect(getByTestId('scan-loading')).toBeTruthy();
    expect(getByText('Skanowanie...')).toBeTruthy();
  });

  it('SS1.4: should display discovered devices and allow connecting', () => {
    mockUseBLE.mockReturnValue({
      ...mockUseBLE(),
      devices: [{ id: 'd1', name: 'Waga Xiaomi' }],
    });
    const { getByText } = render(<ScaleScreen />);
    expect(getByText('Waga Xiaomi')).toBeTruthy();

    fireEvent.press(getByText('Połącz'));
    expect(mockConnect).toHaveBeenCalledWith('d1');
  });

  it('SS1.5: should show connected state and weight reading', () => {
    mockUseBLE.mockReturnValue({
      ...mockUseBLE(),
      connectedDevice: { id: 'd1', name: 'Waga Xiaomi' },
      lastReading: 82.5,
    });
    const { getByText } = render(<ScaleScreen />);
    expect(getByText('Połączono: Waga Xiaomi')).toBeTruthy();
    expect(getByText('82.5 kg')).toBeTruthy();
    expect(getByText('Rozłącz')).toBeTruthy();
  });

  it('SS1.6: should call disconnect when button pressed', () => {
    mockUseBLE.mockReturnValue({
      ...mockUseBLE(),
      connectedDevice: { id: 'd1', name: 'Waga' },
    });
    const { getByText } = render(<ScaleScreen />);
    fireEvent.press(getByText('Rozłącz'));
    expect(mockDisconnect).toHaveBeenCalled();
  });

  it('SS1.7: should subscribe to weight characteristic when connected', () => {
    mockUseBLE.mockReturnValue({
      ...mockUseBLE(),
      connectedDevice: { id: 'd1', name: 'Waga' },
    });
    render(<ScaleScreen />);
    expect(mockOnWeightReading).toHaveBeenCalled();
  });
});
