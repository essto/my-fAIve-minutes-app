import React, { useEffect } from 'react';
import { View, Text, ActivityIndicator, ScrollView } from 'react-native';
import { useHealthData, Anomaly, WeightReading } from '../hooks/useHealthData';
import Animated, { FadeInDown, FadeInUp, FadeInRight } from 'react-native-reanimated';
import { BlurView } from 'expo-blur';

export const HealthScreen = () => {
  const {
    healthScore,
    healthBreakdown,
    weightHistory,
    anomalies,
    isLoading,
    fetchHealthScore,
    fetchWeightHistory,
    fetchAnomalies,
  } = useHealthData();

  useEffect(() => {
    fetchHealthScore();
    fetchWeightHistory(30);
    fetchAnomalies();
  }, []);

  if (isLoading) {
    return (
      <View className="flex-1 bg-background justify-center items-center">
        <ActivityIndicator testID="health-loading" size="large" color="#8251EE" />
      </View>
    );
  }

  const latestWeight =
    weightHistory?.length > 0 ? weightHistory[weightHistory.length - 1].value : null;

  return (
    <ScrollView className="flex-1 bg-background p-6">
      <Animated.Text
        entering={FadeInDown.springify()}
        className="text-3xl font-bold text-foreground mb-8 mt-2"
      >
        Twoje Zdrowie
      </Animated.Text>

      <Animated.View
        entering={FadeInUp.delay(100).springify()}
        className="mb-6 overflow-hidden rounded-3xl border border-border"
      >
        <BlurView intensity={30} tint="dark" className="p-6">
          <Text className="text-foreground font-semibold text-lg mb-4">
            Ogólny Health Score: <Text className="text-brand font-bold">{healthScore}</Text>
          </Text>
          {healthBreakdown && (
            <View className="flex-row justify-between mt-2 pt-4 border-t border-border">
              <View className="items-center">
                <Text className="text-xs text-muted-foreground mb-1 uppercase">Waga</Text>
                <Text className="text-foreground font-medium">{healthBreakdown.weight}</Text>
              </View>
              <View className="items-center">
                <Text className="text-xs text-muted-foreground mb-1 uppercase">Sen</Text>
                <Text className="text-foreground font-medium">{healthBreakdown.sleep}</Text>
              </View>
              <View className="items-center">
                <Text className="text-xs text-muted-foreground mb-1 uppercase">Aktywność</Text>
                <Text className="text-foreground font-medium">{healthBreakdown.activity}</Text>
              </View>
            </View>
          )}
        </BlurView>
      </Animated.View>

      <Animated.View
        entering={FadeInUp.delay(200).springify()}
        className="mb-6 overflow-hidden rounded-3xl border border-border"
      >
        <BlurView intensity={30} tint="dark" className="p-6">
          <Text className="text-secondary-foreground font-medium mb-1 opacity-80 uppercase tracking-widest text-xs">
            Waga
          </Text>
          {latestWeight ? (
            <View className="flex-row items-end gap-2">
              <Text className="text-4xl font-bold text-foreground">{latestWeight}</Text>
              <Text className="text-brand text-xl font-medium mb-1">kg</Text>
            </View>
          ) : (
            <Text className="text-muted-foreground italic">Brak danych</Text>
          )}
        </BlurView>
      </Animated.View>

      {anomalies && anomalies.length > 0 && (
        <Animated.View
          entering={FadeInUp.delay(300).springify()}
          className="mb-8 overflow-hidden rounded-3xl border border-red-500/50 bg-red-500/10"
        >
          <BlurView intensity={40} tint="dark" className="p-6">
            <Text className="text-red-400 font-bold mb-3 flex-row items-center flex">
              ⚠️ Wykryto Anomalie
            </Text>
            {anomalies.map((anom: Anomaly, idx: number) => (
              <Animated.Text
                entering={FadeInRight.delay(400 + idx * 100)}
                key={anom.id}
                className="text-foreground text-sm flex-row mb-2"
              >
                <Text className="text-red-400 font-bold">•</Text> {anom.message}
              </Animated.Text>
            ))}
          </BlurView>
        </Animated.View>
      )}
    </ScrollView>
  );
};
