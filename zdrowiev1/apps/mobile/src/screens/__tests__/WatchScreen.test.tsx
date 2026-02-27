import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { WatchScreen } from '../WatchScreen';
import { useWatchBLE } from '../../hooks/useWatchBLE';

jest.mock('../../hooks/useWatchBLE');
const mockUseWatchBLE = useWatchBLE as jest.Mock;

describe('WatchScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('W1.1: should render scan button initially', () => {
    mockUseWatchBLE.mockReturnValue({
      isScanning: false,
      devices: [],
      connectedDevice: null,
      heartRate: null,
      error: null,
      startScan: jest.fn(),
      connectToDevice: jest.fn(),
      disconnect: jest.fn(),
    });

    const { getByTestId } = render(<WatchScreen />);
    expect(getByTestId('scan-watches-btn')).toBeTruthy();
  });

  it('W1.2: should show discovered devices', () => {
    mockUseWatchBLE.mockReturnValue({
      isScanning: false,
      devices: [
        { id: '1', name: 'Zegarek 1' },
        { id: '2', name: 'Zegarek 2' },
      ],
      connectedDevice: null,
      heartRate: null,
      error: null,
      startScan: jest.fn(),
      connectToDevice: jest.fn(),
      disconnect: jest.fn(),
    });

    const { getAllByTestId } = render(<WatchScreen />);
    expect(getAllByTestId('watch-device')).toHaveLength(2);
  });

  it('W1.3: should show live BPM after connect', () => {
    mockUseWatchBLE.mockReturnValue({
      isScanning: false,
      devices: [],
      connectedDevice: { id: '1', name: 'Zegarek' },
      heartRate: 75,
      error: null,
      startScan: jest.fn(),
      connectToDevice: jest.fn(),
      disconnect: jest.fn(),
    });

    const { getByText } = render(<WatchScreen />);
    expect(getByText('75 BPM')).toBeTruthy();
  });

  it('W1.4: should show error message', () => {
    mockUseWatchBLE.mockReturnValue({
      isScanning: false,
      devices: [],
      connectedDevice: null,
      heartRate: null,
      error: 'Bluetooth wyłączony',
      startScan: jest.fn(),
      connectToDevice: jest.fn(),
      disconnect: jest.fn(),
    });

    const { getByText } = render(<WatchScreen />);
    expect(getByText('Bluetooth wyłączony')).toBeTruthy();
  });

  it('W1.5: should call disconnect when disconnect button is pressed', () => {
    const mockDisconnect = jest.fn();
    mockUseWatchBLE.mockReturnValue({
      isScanning: false,
      devices: [],
      connectedDevice: { id: '1', name: 'Zegarek' },
      heartRate: 75,
      error: null,
      startScan: jest.fn(),
      connectToDevice: jest.fn(),
      disconnect: mockDisconnect,
    });

    const { getByTestId } = render(<WatchScreen />);
    fireEvent.press(getByTestId('disconnect-btn'));
    expect(mockDisconnect).toHaveBeenCalled();
  });
});
