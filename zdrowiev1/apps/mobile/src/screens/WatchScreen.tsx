import React from 'react';
import { View, Text, TouchableOpacity, FlatList, ActivityIndicator } from 'react-native';
import { useWatchBLE } from '../hooks/useWatchBLE';
import { ScreenHeader } from '../components/ScreenHeader';
import { GlassCard } from '../components/GlassCard';

export const WatchScreen = () => {
  const {
    isScanning,
    devices,
    connectedDevice,
    heartRate,
    error,
    startScan,
    connectToDevice,
    disconnect,
  } = useWatchBLE();

  return (
    <View className="flex-1 bg-background p-6">
      <ScreenHeader title="Zegarki i Opaski" />

      {error && (
        <View className="bg-destructive/20 p-4 rounded-xl mb-4">
          <Text className="text-destructive text-center">{error}</Text>
        </View>
      )}

      {connectedDevice ? (
        <GlassCard intensity={20}>
          <Text className="text-foreground font-medium mb-4 text-center">
            Połączono z: {connectedDevice.name || 'Zegarek'}
          </Text>

          <View className="items-center mb-6">
            <Text className="text-xs text-muted-foreground uppercase tracking-widest mb-2">
              Tętno na żywo
            </Text>
            {heartRate !== null ? (
              <Text className="text-5xl font-bold text-brand animate-pulse">
                {heartRate} <Text className="text-2xl text-muted-foreground">BPM</Text>
              </Text>
            ) : (
              <ActivityIndicator color="#0ea5e9" size="large" />
            )}
          </View>

          <TouchableOpacity
            testID="disconnect-btn"
            className="border border-destructive py-3 rounded-xl items-center"
            onPress={disconnect}
          >
            <Text className="text-destructive font-medium">Rozłącz</Text>
          </TouchableOpacity>
        </GlassCard>
      ) : (
        <GlassCard intensity={20}>
          <TouchableOpacity
            testID="scan-watches-btn"
            className="bg-brand py-4 rounded-xl flex-row justify-center items-center mb-6"
            onPress={startScan}
            disabled={isScanning}
          >
            {isScanning ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text className="text-white font-bold text-lg">Skanuj Bluetooth</Text>
            )}
          </TouchableOpacity>

          <Text className="text-foreground font-medium mb-4">Znalezione urządzenia:</Text>

          {devices.length === 0 && !isScanning && (
            <Text className="text-muted-foreground text-center italic py-4">
              Brak urządzeń w pobliżu
            </Text>
          )}

          <FlatList
            data={devices}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <TouchableOpacity
                testID="watch-device"
                className="bg-muted/30 p-4 rounded-xl mb-2 flex-row justify-between items-center"
                onPress={() => connectToDevice(item.id)}
              >
                <Text className="text-foreground font-medium">
                  {item.name || 'Nieznane urządzenie'}
                </Text>
                <Text className="text-brand font-medium text-xs">POŁĄCZ</Text>
              </TouchableOpacity>
            )}
            scrollEnabled={false}
          />
        </GlassCard>
      )}
    </View>
  );
};
