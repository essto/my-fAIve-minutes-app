import React, { useEffect } from 'react';
import { View, Text, ActivityIndicator, ScrollView } from 'react-native';
import { useAuth } from '../hooks/useAuth';
import { useHealthData } from '../hooks/useHealthData';
import { useDietData } from '../hooks/useDietData';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { BlurView } from 'expo-blur';

export const HomeScreen = () => {
  const { user } = useAuth();
  const { healthScore, fetchHealthScore, isLoading: healthLoading } = useHealthData();
  const { dailySummary, fetchDailySummary, isLoading: dietLoading } = useDietData();

  useEffect(() => {
    fetchHealthScore();
    // In a real app, you would pass the current date. Hardcoded for MVP logic here.
    const today = new Date().toISOString().split('T')[0];
    fetchDailySummary(today);
  }, []);

  const isLoading = healthLoading || dietLoading;

  if (isLoading) {
    return (
      <View className="flex-1 bg-background justify-center items-center">
        <ActivityIndicator testID="home-loading" size="large" color="#8251EE" />
      </View>
    );
  }

  return (
    <ScrollView className="flex-1 bg-background p-6">
      <Animated.Text
        entering={FadeInDown.delay(100).springify()}
        className="text-3xl font-bold text-foreground mb-8 mt-2"
      >
        Cześć, {user?.email?.split('@')[0] || 'Użytkowniku'} 👋
      </Animated.Text>

      <Animated.View
        entering={FadeInUp.delay(200).springify()}
        className="mb-6 overflow-hidden rounded-3xl border border-border"
      >
        <BlurView intensity={30} tint="dark" className="p-6">
          <Text className="text-secondary-foreground font-medium mb-2 opacity-80 uppercase tracking-widest text-xs">
            Puls Zdrowia
          </Text>
          <View className="flex-row items-end gap-2">
            <Text className="text-6xl font-bold text-brand">{healthScore ?? '-'}</Text>
            <Text className="text-xl text-foreground font-medium mb-2 opacity-90">/ 100</Text>
          </View>
        </BlurView>
      </Animated.View>

      <Animated.View
        entering={FadeInUp.delay(300).springify()}
        className="mb-8 overflow-hidden rounded-3xl border border-border"
      >
        <BlurView intensity={30} tint="dark" className="p-6">
          <Text className="text-secondary-foreground font-medium mb-2 opacity-80 uppercase tracking-widest text-xs">
            Bilans Dzienny
          </Text>
          <View className="flex-row items-end gap-2">
            <Text className="text-5xl font-bold text-foreground">
              {dailySummary?.total.calories ?? 0}
            </Text>
            <Text className="text-xl text-brand font-medium mb-1">kcal</Text>
          </View>
        </BlurView>
      </Animated.View>
    </ScrollView>
  );
};
