import React, { useEffect } from 'react';
import { View, Text, TouchableOpacity, FlatList, ActivityIndicator } from 'react-native';
import { useBLE } from '../hooks/useBLE';
import { FadeInUp, Layout } from 'react-native-reanimated';
import { GlassCard } from '../components/GlassCard';
import { MetricCard } from '../components/MetricCard';
import { ScreenHeader } from '../components/ScreenHeader';

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
    <View className="flex-1 bg-background p-6">
      <ScreenHeader title="Waga Bluetooth" />

      {error ? (
        <Text testID="ble-error" className="text-destructive font-medium mb-4">
          {error}
        </Text>
      ) : null}

      {connectedDevice ? (
        <GlassCard entering={FadeInUp.springify()} className="items-center p-8">
          <Text className="text-base text-muted-foreground mb-4 text-center">
            Urządzenie:{' '}
            <Text className="font-bold text-foreground">{connectedDevice.name || 'Nieznane'}</Text>
          </Text>

          <MetricCard
            label="Aktualna Waga"
            value={lastReading !== null ? lastReading.toFixed(1) : '--'}
            unit="kg"
          />

          <TouchableOpacity
            className="border border-destructive py-3 px-8 rounded-xl items-center active:bg-destructive/10 w-full mt-6"
            onPress={disconnect}
          >
            <Text className="text-destructive font-bold text-base">Rozłącz</Text>
          </TouchableOpacity>
        </GlassCard>
      ) : (
        <>
          <View className="flex-row items-center mb-6 mt-2">
            <TouchableOpacity
              className={`flex-1 py-4 px-6 rounded-xl items-center flex-row justify-center ${isScanning ? 'bg-brand/50' : 'bg-brand active:bg-brand-hover'}`}
              onPress={startScan}
              disabled={isScanning}
            >
              <Text className="text-white font-bold text-base">
                {isScanning ? 'Skanowanie...' : 'Wyszukaj urządzenie'}
              </Text>
            </TouchableOpacity>

            {isScanning && (
              <ActivityIndicator
                testID="scan-loading"
                className="ml-4"
                size="small"
                color="#8251EE"
              />
            )}
          </View>

          <FlatList
            data={devices}
            keyExtractor={(item) => item.id}
            ListEmptyComponent={
              !isScanning ? (
                <View className="py-10 items-center justify-center border border-dashed border-border rounded-xl mt-4">
                  <Text className="text-muted-foreground text-center text-base">
                    Brak urządzeń w zasięgu
                  </Text>
                </View>
              ) : null
            }
            renderItem={({ item }) => (
              <GlassCard
                layout={Layout.springify()}
                entering={FadeInUp.springify()}
                className="p-5 mb-3 flex-row justify-between items-center"
                intensity={20}
              >
                <Text className="text-foreground font-medium text-lg">
                  {item.name || 'Smart Sacle'}
                </Text>
                <TouchableOpacity
                  className="bg-brand/10 py-2 px-4 rounded-lg active:bg-brand/20"
                  onPress={() => connectToDevice(item.id)}
                >
                  <Text className="text-brand font-bold text-sm">Połącz</Text>
                </TouchableOpacity>
              </GlassCard>
            )}
          />
        </>
      )}
    </View>
  );
};
