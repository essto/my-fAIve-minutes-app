import React, { useEffect } from 'react';
import { ScrollView, View, Text } from 'react-native';
import { useAuth } from '../hooks/useAuth';
import { useHealthData } from '../hooks/useHealthData';
import { useDietData } from '../hooks/useDietData';
import { FadeInUp } from 'react-native-reanimated';
import { GlassCard } from '../components/GlassCard';
import { MetricCard } from '../components/MetricCard';
import { ScreenHeader } from '../components/ScreenHeader';
import { LoadingScreen } from '../components/LoadingScreen';
import { HealthScoreRing } from '../components/HealthScoreRing';
import { ActivityRings } from '../components/ActivityRings';
import { SparklineChart } from '../components/SparklineChart';
import { NotificationBell } from '../components/NotificationBell';

export const HomeScreen = () => {
  const { user } = useAuth();
  const { healthScore, fetchHealthScore, isLoading: healthLoading } = useHealthData();
  const { dailySummary, fetchDailySummary, isLoading: dietLoading } = useDietData();

  useEffect(() => {
    fetchHealthScore();
    const today = new Date().toISOString().split('T')[0];
    fetchDailySummary(today);
  }, []);

  const isLoading = healthLoading || dietLoading;

  if (isLoading) {
    return <LoadingScreen testID="home-loading" />;
  }

  return (
    <ScrollView className="flex-1 bg-background p-6">
      <View className="flex-row justify-between items-baseline mb-6 mt-2">
        <ScreenHeader title={`Cześć, ${user?.email?.split('@')[0] || 'Użytkowniku'} 👋`} />
        <NotificationBell count={3} />
      </View>

      <GlassCard
        testID="health-score-ring"
        entering={FadeInUp.delay(200).springify()}
        className="mb-6 items-center py-6"
      >
        <HealthScoreRing score={healthScore !== null ? healthScore : 0} />
      </GlassCard>

      <GlassCard entering={FadeInUp.delay(300).springify()} className="mb-6">
        <MetricCard label="Bilans Dzienny" value={dailySummary?.total.calories ?? 0} unit="kcal" />
        <View className="items-center mt-2">
          <SparklineChart
            data={[
              { value: 1800, date: '1' },
              { value: 2100, date: '2' },
              { value: 1950, date: '3' },
              { value: 2200, date: '4' },
              { value: dailySummary?.total.calories || 2000, date: '5' },
            ]}
            color="#10B981"
          />
        </View>
      </GlassCard>

      <GlassCard entering={FadeInUp.delay(400).springify()} className="mb-12 items-center py-6">
        <Text className="text-muted-foreground font-semibold text-sm mb-4 uppercase tracking-widest self-start w-full px-4">
          Twoja Aktywność
        </Text>
        <ActivityRings stepsProgress={0.7} caloriesProgress={0.5} exerciseProgress={0.3} />
      </GlassCard>
    </ScrollView>
  );
};
