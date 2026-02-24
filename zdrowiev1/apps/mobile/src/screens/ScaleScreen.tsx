import React, { useEffect } from 'react';
import { View, Text, TouchableOpacity, FlatList, ActivityIndicator } from 'react-native';
import { useBLE } from '../hooks/useBLE';
import Animated, { FadeInDown, FadeInUp, Layout } from 'react-native-reanimated';
import { BlurView } from 'expo-blur';

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
    <View className="flex-1 bg-background p-6 pt-10">
      <Animated.Text
        entering={FadeInDown.springify()}
        className="text-3xl font-bold text-foreground mb-6"
      >
        Waga Bluetooth
      </Animated.Text>

      {error ? <Text className="text-destructive font-medium mb-4">{error}</Text> : null}

      {connectedDevice ? (
        <Animated.View
          entering={FadeInUp.springify()}
          className="overflow-hidden rounded-3xl border border-border mt-4"
        >
          <BlurView intensity={30} tint="dark" className="p-8 items-center">
            <Text className="text-base text-muted-foreground mb-2 text-center">
              Połączono:{' '}
              <Text className="font-bold text-foreground">
                {connectedDevice.name || 'Nieznane urządzenie'}
              </Text>
            </Text>
            <Text className="text-[56px] font-bold text-brand my-6">
              {lastReading !== null ? `${lastReading.toFixed(1)} kg` : '-- kg'}
            </Text>
            <TouchableOpacity
              className="border border-destructive py-3 px-8 rounded-xl items-center active:bg-destructive/10 w-full"
              onPress={disconnect}
            >
              <Text className="text-destructive font-bold text-base">Rozłącz urządzenie</Text>
            </TouchableOpacity>
          </BlurView>
        </Animated.View>
      ) : (
        <>
          <Animated.View
            entering={FadeInUp.delay(100).springify()}
            className="flex-row items-center mb-6 mt-2"
          >
            <TouchableOpacity
              className={`flex-1 py-4 px-6 rounded-xl items-center flex-row justify-center ${isScanning ? 'bg-brand/50' : 'bg-brand active:bg-brand-hover'}`}
              onPress={startScan}
              disabled={isScanning}
            >
              <Text className="text-white font-bold text-base">
                {isScanning ? 'Skanowanie sieci...' : 'Wyszukaj urządzenie'}
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
          </Animated.View>

          <FlatList
            data={devices}
            keyExtractor={(item) => item.id}
            ListEmptyComponent={
              !isScanning ? (
                <View className="py-10 items-center justify-center border border-dashed border-border rounded-xl mt-4">
                  <Text className="text-muted-foreground text-center text-base">
                    Brak zgodnych urządzeń w zasięgu
                  </Text>
                </View>
              ) : null
            }
            renderItem={({ item }) => (
              <Animated.View
                layout={Layout.springify()}
                entering={FadeInUp.springify()}
                className="bg-card p-5 rounded-2xl mb-3 flex-row justify-between items-center border border-border"
              >
                <Text className="text-foreground font-medium text-lg">
                  {item.name || 'Zegarek / Opaska'}
                </Text>
                <TouchableOpacity
                  className="bg-brand/10 py-2 px-4 rounded-lg active:bg-brand/20"
                  onPress={() => connectToDevice(item.id)}
                >
                  <Text className="text-brand font-bold text-sm">Połącz</Text>
                </TouchableOpacity>
              </Animated.View>
            )}
          />
        </>
      )}
    </View>
  );
};
