import React, { useEffect } from 'react';
import { ScrollView } from 'react-native';
import { useAuth } from '../hooks/useAuth';
import { useHealthData } from '../hooks/useHealthData';
import { useDietData } from '../hooks/useDietData';
import { FadeInUp } from 'react-native-reanimated';
import { GlassCard } from '../components/GlassCard';
import { MetricCard } from '../components/MetricCard';
import { ScreenHeader } from '../components/ScreenHeader';
import { LoadingScreen } from '../components/LoadingScreen';

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
      <ScreenHeader title={`Cześć, ${user?.email?.split('@')[0] || 'Użytkowniku'} 👋`} />

      <GlassCard entering={FadeInUp.delay(200).springify()}>
        <MetricCard label="Puls Zdrowia" value={healthScore ?? '-'} unit="/ 100" />
      </GlassCard>

      <GlassCard entering={FadeInUp.delay(300).springify()}>
        <MetricCard label="Bilans Dzienny" value={dailySummary?.total.calories ?? 0} unit="kcal" />
      </GlassCard>
    </ScrollView>
  );
};
