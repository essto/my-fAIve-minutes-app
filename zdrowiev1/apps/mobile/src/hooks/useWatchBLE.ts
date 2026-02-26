import { useState, useEffect, useRef } from 'react';
import { BleManager, Device, BleError } from 'react-native-ble-plx';
import { Buffer } from 'buffer';

const HR_SERVICE_UUID = '180D';
const HR_CHARACTERISTIC_UUID = '2A37';

export const useWatchBLE = () => {
  const [manager] = useState(() => new BleManager());
  const [isScanning, setIsScanning] = useState(false);
  const [devices, setDevices] = useState<Device[]>([]);
  const [connectedDevice, setConnectedDevice] = useState<Device | null>(null);
  const [heartRate, setHeartRate] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const scanTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      manager.destroy();
      if (scanTimeoutRef.current) {
        clearTimeout(scanTimeoutRef.current);
      }
    };
  }, [manager]);

  const startScan = async () => {
    setError(null);
    setDevices([]);

    try {
      const state = await manager.state();
      if (state !== 'PoweredOn') {
        setError('Bluetooth jest wyłączony lub brak uprawnień.');
        return;
      }

      setIsScanning(true);

      manager.startDeviceScan(null, null, (bleError: BleError | null, device: Device | null) => {
        if (bleError) {
          setError(bleError.message);
          setIsScanning(false);
          return;
        }

        if (device && device.name) {
          setDevices((prev) => {
            if (!prev.find((d) => d.id === device.id)) {
              return [...prev, device];
            }
            return prev;
          });
        }
      });

      scanTimeoutRef.current = setTimeout(() => {
        manager.stopDeviceScan();
        setIsScanning(false);
      }, 10000); // 10 seconds scan
    } catch (e: any) {
      setError(e.message || 'Nieznany błąd podczas skanowania');
      setIsScanning(false);
    }
  };

  const connectToDevice = async (deviceId: string) => {
    try {
      manager.stopDeviceScan();
      setIsScanning(false);
      if (scanTimeoutRef.current) clearTimeout(scanTimeoutRef.current);

      const device = await manager.connectToDevice(deviceId);
      await device.discoverAllServicesAndCharacteristics();
      setConnectedDevice(device);

      // Start monitoring HR characteristic
      manager.monitorCharacteristicForDevice(
        device.id,
        HR_SERVICE_UUID,
        HR_CHARACTERISTIC_UUID,
        (bleError: BleError | null, characteristic: any) => {
          if (bleError || !characteristic?.value) {
            return;
          }

          const decodedValue = Buffer.from(characteristic.value, 'base64');
          const flags = decodedValue[0];

          // Check if HR format is 8-bit or 16-bit
          // The 0th bit of flags byte tells us the format: 0 for 8-bit, 1 for 16-bit
          const is16Bit = (flags & 0x01) !== 0;

          let currentHeartRate;
          if (is16Bit) {
            currentHeartRate = decodedValue.readUInt16LE(1);
          } else {
            currentHeartRate = decodedValue.readUInt8(1);
          }

          setHeartRate(currentHeartRate);
        },
      );
    } catch (e: any) {
      setError(e.message || 'Nie udało się połączyć z urządzeniem.');
      setConnectedDevice(null);
    }
  };

  const disconnect = async () => {
    if (connectedDevice) {
      try {
        await manager.cancelDeviceConnection(connectedDevice.id);
        setConnectedDevice(null);
        setHeartRate(null);
      } catch (e: any) {
        setError(e.message);
      }
    }
  };

  return {
    isScanning,
    devices,
    connectedDevice,
    heartRate,
    startScan,
    connectToDevice,
    disconnect,
    error,
  };
};
