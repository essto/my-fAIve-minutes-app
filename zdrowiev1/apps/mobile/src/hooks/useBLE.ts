import { useState, useEffect, useRef } from 'react';
import { BleManager, Device, BleError } from 'react-native-ble-plx';
import { Buffer } from 'buffer'; // Need a polyfill if not bare RN

// For Expo, usually we'd maintain the manager instance via ref or outside the hook,
// but for the sake of simplicity and adhering to typical hooks pattern:
let manager: BleManager;

export const useBLE = () => {
  const [devices, setDevices] = useState<Device[]>([]);
  const [connectedDevice, setConnectedDevice] = useState<Device | null>(null);
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [lastReading, setLastReading] = useState<number | null>(null);

  const scanTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Only initialize once
    if (!manager) {
      manager = new BleManager();
    }

    return () => {
      // Typically we don't destroy it here if it's singleton across the app,
      // but to match T4.10 requirement for complete clean up in unmount context:
      manager.destroy();
      // Need to unset so we can recreate if the hook mounts again
      manager = undefined as any;
    };
  }, []);

  const startScan = async () => {
    setError(null);
    setDevices([]);

    try {
      const state = await manager.state();

      // In iOS/Android real devices, we also need to explicitly ask for Permissions
      // using expo-permissions or equivalent. Here we just assert the module state.
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
            const exists = prev.find((p) => p.id === device.id);
            if (!exists) return [...prev, device];
            return prev;
          });
        }
      });

      // Stop scanning automatically after 10s
      scanTimeoutRef.current = setTimeout(() => {
        if (manager) manager.stopDeviceScan();
        setIsScanning(false);
      }, 10000);
    } catch (err: any) {
      setError(err.message || 'Wystąpił nieoczekiwany błąd');
      setIsScanning(false);
    }
  };

  const connectToDevice = async (deviceId: string) => {
    try {
      setError(null);
      // Wait for any existing scan to stop
      if (isScanning) {
        manager.stopDeviceScan();
        if (scanTimeoutRef.current) clearTimeout(scanTimeoutRef.current);
        setIsScanning(false);
      }

      const device = await manager.connectToDevice(deviceId);
      await device.discoverAllServicesAndCharacteristics();

      setConnectedDevice(device);
    } catch (err: any) {
      setError(err.message || 'Nie udało się połączyć');
    }
  };

  const onWeightReading = (callback: (weight: number) => void) => {
    if (!connectedDevice) return;

    // Weight scale characteristic format depends on the scale vendor (e.g. Xiaomi, Withings)
    // UUIDs mock values
    const serviceUUID = '181D'; // Weight Scale Service standard UUID
    const characteristicUUID = '2A9D'; // Weight Measurement

    const subscription = manager.monitorCharacteristicForDevice(
      connectedDevice.id,
      serviceUUID,
      characteristicUUID,
      (err, characteristic) => {
        if (err) return;
        if (characteristic?.value) {
          // Decode value. Example: Buffer.from(val, 'base64') -> parse number.
          const buffer = Buffer.from(characteristic.value, 'base64');
          const weightStr = buffer.toString('utf-8');
          const weight = parseFloat(weightStr);

          if (!isNaN(weight)) {
            setLastReading(weight);
            callback(weight);
          }
        }
      },
    );

    return () => subscription.remove();
  };

  const disconnect = async () => {
    if (connectedDevice) {
      await manager.cancelDeviceConnection(connectedDevice.id);
      setConnectedDevice(null);
    }
  };

  return {
    devices,
    connectedDevice,
    isScanning,
    error,
    lastReading,
    startScan,
    connectToDevice,
    onWeightReading,
    disconnect,
  };
};
