import React, { useEffect } from 'react';
import { View, Text, ScrollView } from 'react-native';
import { useHealthData, Anomaly } from '../hooks/useHealthData';
import { FadeInUp, FadeInRight } from 'react-native-reanimated';
import { GlassCard } from '../components/GlassCard';
import { MetricCard } from '../components/MetricCard';
import { ScreenHeader } from '../components/ScreenHeader';
import { LoadingScreen } from '../components/LoadingScreen';

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
    return <LoadingScreen testID="health-loading" />;
  }

  const latestWeight =
    weightHistory?.length > 0 ? weightHistory[weightHistory.length - 1].value : null;

  return (
    <ScrollView className="flex-1 bg-background p-6">
      <ScreenHeader title="Twoje Zdrowie" />

      <GlassCard entering={FadeInUp.delay(100).springify()}>
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
      </GlassCard>

      <GlassCard entering={FadeInUp.delay(200).springify()}>
        <MetricCard label="Waga" value={latestWeight ?? '-'} unit="kg" />
        {!latestWeight && <Text className="text-muted-foreground italic">Brak danych</Text>}
      </GlassCard>

      {anomalies && anomalies.length > 0 && (
        <GlassCard
          entering={FadeInUp.delay(300).springify()}
          className="bg-red-500/10 border-red-500/50"
          intensity={40}
        >
          <Text className="text-red-400 font-bold mb-3">⚠️ Wykryto Anomalie</Text>
          {anomalies.map((anom: Anomaly, idx: number) => (
            <Text key={anom.id} className="text-foreground text-sm mb-2">
              <Text className="text-red-400 font-bold">•</Text> {anom.message}
            </Text>
          ))}
        </GlassCard>
      )}
    </ScrollView>
  );
};
