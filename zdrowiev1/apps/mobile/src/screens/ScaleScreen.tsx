import React, { useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  ActivityIndicator,
} from 'react-native';
import { useBLE } from '../hooks/useBLE';

export const ScaleScreen = () => {
  const {
    devices,
    connectedDevice,
    isScanning,
    error,
    lastReading,
    startScan,
    connectToDevice,
    disconnect,
    onWeightReading,
  } = useBLE();

  // SS1.7: Auto-subscribe to characteristic when device is connected
  useEffect(() => {
    let unsubscribe: (() => void) | undefined;
    if (connectedDevice) {
      unsubscribe = onWeightReading((weight) => {
        // Dodatkowa logika po odebraniu wagi, np. zapis do AsyncStorage lub wysłanie na backend
        console.log(`Odczytana waga: ${weight} kg`);
      });
    }

    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, [connectedDevice, onWeightReading]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Waga Bluetooth</Text>

      {error && <Text style={styles.errorText}>{error}</Text>}

      {connectedDevice ? (
        <View style={styles.connectedCard}>
          <Text style={styles.subtitle}>
            Połączono: {connectedDevice.name || 'Nieznane urządzenie'}
          </Text>
          <Text style={styles.readingText}>
            {lastReading !== null ? `${lastReading.toFixed(1)} kg` : '-- kg'}
          </Text>
          <TouchableOpacity style={styles.buttonSecondary} onPress={disconnect}>
            <Text style={styles.buttonTextSecondary}>Rozłącz</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <>
          <View style={styles.scanSection}>
            <TouchableOpacity
              style={[styles.buttonPrimary, isScanning && styles.buttonDisabled]}
              onPress={startScan}
              disabled={isScanning}
            >
              <Text style={styles.buttonTextPrimary}>
                {isScanning ? 'Skanowanie...' : 'Wyszukaj urządzenie'}
              </Text>
            </TouchableOpacity>

            {isScanning && (
              <ActivityIndicator
                testID="scan-loading"
                style={styles.loader}
                size="small"
                color="#007AFF"
              />
            )}
          </View>

          <FlatList
            data={devices}
            keyExtractor={(item) => item.id}
            ListEmptyComponent={
              !isScanning ? <Text style={styles.emptyText}>Brak urządzeń w pobliżu</Text> : null
            }
            renderItem={({ item }) => (
              <View style={styles.deviceItem}>
                <Text style={styles.deviceName}>{item.name || 'Nieznane urządzenie'}</Text>
                <TouchableOpacity
                  style={styles.connectButton}
                  onPress={() => connectToDevice(item.id)}
                >
                  <Text style={styles.connectButtonText}>Połącz</Text>
                </TouchableOpacity>
              </View>
            )}
          />
        </>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FAF9F6',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1A1A1A',
    marginBottom: 20,
    marginTop: 20,
  },
  subtitle: {
    fontSize: 18,
    color: '#333',
    marginBottom: 10,
  },
  errorText: {
    color: '#D32F2F',
    marginBottom: 10,
  },
  connectedCard: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 5,
    elevation: 2,
    marginTop: 20,
  },
  readingText: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#007AFF',
    marginVertical: 20,
  },
  scanSection: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  buttonPrimary: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
    flex: 1,
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: '#A9CFFF',
  },
  buttonTextPrimary: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  buttonSecondary: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#FF3B30',
    paddingVertical: 12,
    paddingHorizontal: 40,
    borderRadius: 12,
    width: '100%',
    alignItems: 'center',
  },
  buttonTextSecondary: {
    color: '#FF3B30',
    fontSize: 16,
    fontWeight: '600',
  },
  loader: {
    marginLeft: 15,
  },
  emptyText: {
    color: '#666',
    textAlign: 'center',
    marginTop: 20,
  },
  deviceItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
  },
  deviceName: {
    fontSize: 16,
    color: '#1A1A1A',
  },
  connectButton: {
    backgroundColor: '#E5F1FF',
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderRadius: 8,
  },
  connectButtonText: {
    color: '#007AFF',
    fontWeight: '600',
  },
});
