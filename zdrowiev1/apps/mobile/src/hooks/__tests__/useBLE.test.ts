import { renderHook, act, waitFor } from '@testing-library/react-native';
import { useBLE } from '../useBLE';
import { BleManager } from 'react-native-ble-plx';

jest.mock('react-native-ble-plx', () => {
  const mockBleManager = {
    state: jest.fn(),
    startDeviceScan: jest.fn(),
    stopDeviceScan: jest.fn(),
    connectToDevice: jest.fn(),
    discoverAllServicesAndCharacteristics: jest.fn(),
    monitorCharacteristicForDevice: jest.fn(),
    cancelDeviceConnection: jest.fn(),
    destroy: jest.fn(),
  };
  return {
    BleManager: jest.fn(() => mockBleManager),
    // Export instance for checking calls
    _mockBleManager: mockBleManager,
  };
});

describe('useBLE', () => {
  let mockBleManager: any;

  beforeEach(() => {
    jest.clearAllMocks();
    mockBleManager = require('react-native-ble-plx')._mockBleManager;
  });

  // ----- T4.1 Inicjalizacja managera -----
  it('T4.1: should initialize BleManager on mount', () => {
    renderHook(() => useBLE());
    expect(BleManager).toHaveBeenCalledTimes(1);
  });

  // ----- T4.2 Sprawdzenie uprawnień Bluetooth -----
  it('T4.2: should check Bluetooth state before scanning', async () => {
    mockBleManager.state.mockResolvedValue('PoweredOn');
    const { result } = renderHook(() => useBLE());
    await act(async () => {
      await result.current.startScan();
    });
    expect(mockBleManager.state).toHaveBeenCalled();
  });

  // ----- T4.3 Skanowanie — BT wyłączony -----
  it('T4.3: should return error when Bluetooth is off', async () => {
    mockBleManager.state.mockResolvedValue('PoweredOff');
    const { result } = renderHook(() => useBLE());
    await act(async () => {
      await result.current.startScan();
    });
    expect(result.current.error).toBe('Bluetooth jest wyłączony lub brak uprawnień.');
    expect(result.current.isScanning).toBe(false);
  });

  // ----- T4.4 Skanowanie — znalezione urządzenia -----
  it('T4.4: should discover BLE devices during scan', async () => {
    mockBleManager.state.mockResolvedValue('PoweredOn');
    mockBleManager.startDeviceScan.mockImplementation((_: any, __: any, callback: any) => {
      // Simulate discovering two devices
      callback(null, { id: 'dev1', name: 'Xiaomi Scale' });
      callback(null, { id: 'dev2', name: 'Withings WBS12' });
    });
    const { result } = renderHook(() => useBLE());

    await act(async () => {
      await result.current.startScan();
    });

    await waitFor(() => {
      expect(result.current.devices).toHaveLength(2);
      expect(result.current.devices[0].name).toBe('Xiaomi Scale');
    });
  });

  // ----- T4.5 Skanowanie — timeout 10s -----
  it('T4.5: should stop scanning after timeout', async () => {
    jest.useFakeTimers();
    mockBleManager.state.mockResolvedValue('PoweredOn');
    mockBleManager.startDeviceScan.mockImplementation(() => {});

    const { result } = renderHook(() => useBLE());

    await act(async () => {
      await result.current.startScan();
    });

    // Scan should be running now
    expect(result.current.isScanning).toBe(true);

    // Fast-forward 10s
    await act(async () => {
      jest.advanceTimersByTime(11000);
    });

    expect(mockBleManager.stopDeviceScan).toHaveBeenCalled();
    expect(result.current.isScanning).toBe(false);
    jest.useRealTimers();
  });

  // ----- T4.6 Parowanie z urządzeniem -----
  it('T4.6: should connect to a BLE device', async () => {
    const mockDevice = {
      id: 'dev1',
      connect: jest
        .fn()
        .mockResolvedValue({
          id: 'dev1',
          discoverAllServicesAndCharacteristics: jest.fn().mockResolvedValue({}),
        }),
      discoverAllServicesAndCharacteristics: jest.fn().mockResolvedValue({}),
    };
    mockBleManager.connectToDevice.mockResolvedValue(mockDevice);

    const { result } = renderHook(() => useBLE());

    await act(async () => {
      await result.current.connectToDevice('dev1');
    });

    await waitFor(() => {
      expect(result.current.connectedDevice).toEqual(expect.objectContaining({ id: 'dev1' }));
    });
  });

  // ----- T4.7 Parowanie — urządzenie poza zasięgiem -----
  it('T4.7: should handle connection failure (device out of range)', async () => {
    mockBleManager.connectToDevice.mockRejectedValue(new Error('Connection timeout'));

    const { result } = renderHook(() => useBLE());

    await act(async () => {
      await result.current.connectToDevice('dev_far');
    });

    expect(result.current.error).toContain('Connection timeout');
    expect(result.current.connectedDevice).toBeNull();
  });

  // ----- T4.8 Odczyt wagi po parowaniu -----
  it('T4.8: should receive weight reading from connected scale', async () => {
    const mockCallback = jest.fn();
    mockBleManager.monitorCharacteristicForDevice.mockImplementation(
      (deviceId: string, serviceUUID: string, charUUID: string, callback: any) => {
        const value = Buffer.from('82.3').toString('base64');
        callback(null, { value });
        return { remove: jest.fn() };
      },
    );

    const { result } = renderHook(() => useBLE());

    const mockDevice = {
      id: 'dev1',
      connect: jest
        .fn()
        .mockResolvedValue({
          id: 'dev1',
          discoverAllServicesAndCharacteristics: jest.fn().mockResolvedValue({}),
        }),
      discoverAllServicesAndCharacteristics: jest.fn().mockResolvedValue({}),
    };
    mockBleManager.connectToDevice.mockResolvedValue(mockDevice);

    await act(async () => {
      await result.current.connectToDevice('dev1');
    });

    await act(async () => {
      result.current.onWeightReading(mockCallback);
    });

    await waitFor(() => {
      expect(result.current.lastReading).toBe(82.3);
      expect(mockCallback).toHaveBeenCalledWith(82.3);
    });
  });

  // ----- T4.9 Rozłączanie -----
  it('T4.9: should disconnect from device and clear state', async () => {
    mockBleManager.cancelDeviceConnection.mockResolvedValue({});
    const { result } = renderHook(() => useBLE());

    const mockDevice = {
      id: 'dev1',
      connect: jest
        .fn()
        .mockResolvedValue({
          id: 'dev1',
          discoverAllServicesAndCharacteristics: jest.fn().mockResolvedValue({}),
        }),
      discoverAllServicesAndCharacteristics: jest.fn().mockResolvedValue({}),
    };
    mockBleManager.connectToDevice.mockResolvedValue(mockDevice);

    await act(async () => {
      await result.current.connectToDevice('dev1');
    });

    await act(async () => {
      await result.current.disconnect();
    });

    expect(mockBleManager.cancelDeviceConnection).toHaveBeenCalledWith('dev1');
    expect(result.current.connectedDevice).toBeNull();
  });

  // ----- T4.10 Cleanup przy unmount -----
  it('T4.10: should cleanup BLE subscriptions on unmount', () => {
    const { unmount } = renderHook(() => useBLE());
    unmount();
    expect(mockBleManager.destroy).toHaveBeenCalled();
  });
});
