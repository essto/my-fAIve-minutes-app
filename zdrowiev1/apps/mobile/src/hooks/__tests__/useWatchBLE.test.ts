import { renderHook, act, waitFor } from '@testing-library/react-native';
import { useWatchBLE } from '../useWatchBLE';
import { BleManager } from 'react-native-ble-plx';

jest.mock('react-native-ble-plx', () => {
  const mockBleManager = {
    state: jest.fn(),
    startDeviceScan: jest.fn(),
    stopDeviceScan: jest.fn(),
    connectToDevice: jest.fn(),
    cancelDeviceConnection: jest.fn(),
    destroy: jest.fn(),
    monitorCharacteristicForDevice: jest.fn(),
  };
  return {
    BleManager: jest.fn(() => mockBleManager),
    _mockBleManager: mockBleManager,
  };
});

describe('useWatchBLE', () => {
  let mockBleManager: any;

  beforeEach(() => {
    jest.clearAllMocks();
    mockBleManager = require('react-native-ble-plx')._mockBleManager;
  });

  it('P3.1: should initialize BleManager on mount', () => {
    renderHook(() => useWatchBLE());
    expect(BleManager).toHaveBeenCalledTimes(1);
  });

  it('P3.2: should discover BLE devices during scan', async () => {
    mockBleManager.state.mockResolvedValue('PoweredOn');
    mockBleManager.startDeviceScan.mockImplementation((_: any, __: any, callback: any) => {
      callback(null, { id: 'w1', name: 'Xiaomi Band 7' });
    });

    const { result } = renderHook(() => useWatchBLE());

    await act(async () => {
      await result.current.startScan();
    });

    await waitFor(() => {
      expect(result.current.devices).toHaveLength(1);
      expect(result.current.devices[0].name).toBe('Xiaomi Band 7');
    });
  });

  it('P3.3: should receive heart rate reading from connected watch', async () => {
    const mockDevice = {
      id: 'w1',
      discoverAllServicesAndCharacteristics: jest.fn().mockResolvedValue({}),
    };
    mockBleManager.connectToDevice.mockResolvedValue(mockDevice);

    const { result } = renderHook(() => useWatchBLE());

    await act(async () => {
      await result.current.connectToDevice('w1');
    });

    // Simulate reading HR data: format is Flags (1 byte) + HR (1 byte if flag 0)
    // 0x00 flag means 8-bit Heart Rate format. 0x48 = 72 BPM.
    const hrBuffer = Buffer.from([0x00, 0x48]).toString('base64');

    act(() => {
      const monitorCallback = mockBleManager.monitorCharacteristicForDevice.mock.calls[0][3];
      monitorCallback(null, { value: hrBuffer });
    });

    expect(result.current.heartRate).toBe(72);
  });
});
