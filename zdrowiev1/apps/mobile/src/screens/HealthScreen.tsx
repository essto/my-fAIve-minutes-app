import React, { useEffect } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, ScrollView } from 'react-native';
import { useHealthData, Anomaly, WeightReading } from '../hooks/useHealthData';

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
      <View style={[styles.container, styles.center]}>
        <ActivityIndicator testID="health-loading" size="large" color="#007AFF" />
      </View>
    );
  }

  const latestWeight =
    weightHistory?.length > 0 ? weightHistory[weightHistory.length - 1].value : null;

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Twoje Zdrowie</Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Ogólny Health Score: {healthScore}</Text>
        {healthBreakdown && (
          <View style={styles.breakdownRow}>
            <Text style={styles.breakdownText}>Waga: {healthBreakdown.weight}</Text>
            <Text style={styles.breakdownText}>Sen: {healthBreakdown.sleep}</Text>
            <Text style={styles.breakdownText}>Aktywność: {healthBreakdown.activity}</Text>
          </View>
        )}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Waga</Text>
        {latestWeight ? (
          <Text style={styles.cardValue}>Ostatni pomiar: {latestWeight} kg</Text>
        ) : (
          <Text style={styles.cardValue}>Brak danych</Text>
        )}
      </View>

      {anomalies && anomalies.length > 0 && (
        <View style={[styles.card, styles.anomalyCard]}>
          <Text style={[styles.cardTitle, styles.anomalyTitle]}>Wykryto Anomalie</Text>
          {anomalies.map((anom: Anomaly) => (
            <Text key={anom.id} style={styles.anomalyText}>
              {`• ${anom.message}`}
            </Text>
          ))}
        </View>
      )}
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
  title: {
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
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 10,
  },
  cardValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#007AFF',
  },
  breakdownRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
  },
  breakdownText: {
    fontSize: 14,
    color: '#666',
  },
  anomalyCard: {
    backgroundColor: '#FFF0F0',
    borderColor: '#FFCCCC',
    borderWidth: 1,
  },
  anomalyTitle: {
    color: '#D32F2F',
  },
  anomalyText: {
    fontSize: 14,
    color: '#B71C1C',
    marginTop: 5,
  },
});
