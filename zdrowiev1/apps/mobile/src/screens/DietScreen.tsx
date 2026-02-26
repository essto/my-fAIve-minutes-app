import React, { useEffect, useState } from 'react';
import { View, Text, ScrollView, TextInput, TouchableOpacity } from 'react-native';
import { useDietData, Meal } from '../hooks/useDietData';
import { FadeInUp } from 'react-native-reanimated';
import { GlassCard } from '../components/GlassCard';
import { MetricCard } from '../components/MetricCard';
import { ScreenHeader } from '../components/ScreenHeader';
import { LoadingScreen } from '../components/LoadingScreen';

export const DietScreen = () => {
  const { meals, dailySummary, isLoading, fetchMeals, fetchDailySummary, logMeal } = useDietData();
  const [newMealName, setNewMealName] = useState('');

  useEffect(() => {
    const today = new Date().toISOString().split('T')[0];
    fetchMeals(today);
    fetchDailySummary(today);
  }, []);

  const handleAddMeal = () => {
    if (newMealName.trim() === '') return;

    logMeal({
      name: newMealName,
      products: [
        {
          name: 'Skanowany produkt (MVP)',
          calories: 300,
          protein: 10,
          carbs: 40,
          fat: 5,
          quantity: 100,
        },
      ],
    });
    setNewMealName('');
  };

  if (isLoading && meals.length === 0) {
    return <LoadingScreen testID="diet-loading" />;
  }

  return (
    <ScrollView className="flex-1 bg-background p-6">
      <ScreenHeader title="Twoja Dieta" />

      <GlassCard entering={FadeInUp.delay(100).springify()}>
        <MetricCard
          label="Podsumowanie Dnia"
          value={dailySummary?.total.calories ?? 0}
          unit="kcal"
        />
        {dailySummary && (
          <View className="flex-row justify-between mt-4 pt-4 border-t border-border">
            <View className="items-center">
              <Text className="text-xs text-muted-foreground mb-1 uppercase">Białko</Text>
              <Text className="text-foreground font-medium">{dailySummary.total.protein}g</Text>
            </View>
            <View className="items-center">
              <Text className="text-xs text-muted-foreground mb-1 uppercase">Węgle.</Text>
              <Text className="text-foreground font-medium">{dailySummary.total.carbs}g</Text>
            </View>
            <View className="items-center">
              <Text className="text-xs text-muted-foreground mb-1 uppercase">Tłuszcze</Text>
              <Text className="text-foreground font-medium">{dailySummary.total.fat}g</Text>
            </View>
          </View>
        )}
      </GlassCard>

      <GlassCard entering={FadeInUp.delay(200).springify()} intensity={20}>
        <Text className="text-foreground font-bold text-lg mb-4">Dodaj Posiłek</Text>
        <TextInput
          className="bg-background border border-border rounded-xl p-4 text-foreground mb-4"
          placeholder="Nazwa posiłku (np. Obiad)"
          placeholderTextColor="#666"
          value={newMealName}
          onChangeText={setNewMealName}
        />
        <TouchableOpacity
          className="bg-brand py-4 rounded-xl items-center flex-row justify-center active:bg-brand-hover mb-2"
          onPress={handleAddMeal}
          disabled={isLoading}
        >
          <Text className="text-white font-bold">Dodaj posiłek</Text>
        </TouchableOpacity>
      </GlassCard>

      <GlassCard entering={FadeInUp.delay(300).springify()} intensity={20}>
        <Text className="text-foreground font-bold text-lg mb-4">Dzisiejsze Posiłki</Text>
        {meals.length === 0 ? (
          <Text className="text-muted-foreground italic">Brak posiłków</Text>
        ) : (
          <View>
            {meals.map((meal: Meal, idx: number) => (
              <View
                key={meal.id}
                className={`py-4 ${idx !== meals.length - 1 ? 'border-b border-border' : ''}`}
              >
                <Text className="text-foreground font-medium text-lg">{meal.name}</Text>
                <Text className="text-muted-foreground text-xs mt-1">
                  Skanowany produkt (MVP) • 300 kcal
                </Text>
              </View>
            ))}
          </View>
        )}
      </GlassCard>
    </ScrollView>
  );
};
