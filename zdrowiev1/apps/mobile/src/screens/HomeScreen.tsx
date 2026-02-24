import React, { useEffect } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, ScrollView } from 'react-native';
import { useAuth } from '../hooks/useAuth';
import { useHealthData } from '../hooks/useHealthData';
import { useDietData } from '../hooks/useDietData';

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
      <View style={[styles.container, styles.center]}>
        <ActivityIndicator testID="home-loading" size="large" color="#007AFF" />
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.greeting}>Cześć, {user?.email}</Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Podsumowanie Zdrowia</Text>
        <Text style={styles.cardValue}>Health Score: {healthScore ?? '-'}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Dieta Dzisiaj</Text>
        <Text style={styles.cardValue}>
          Spożyte kalorie: {dailySummary?.total.calories ?? 0} kcal
        </Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FAF9F6',
    padding: 20,
  },
  center: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  greeting: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#1A1A1A',
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 20,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 5,
    elevation: 2,
  },
  cardTitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 10,
  },
  cardValue: {
    fontSize: 24,
    fontWeight: '600',
    color: '#007AFF',
  },
});
