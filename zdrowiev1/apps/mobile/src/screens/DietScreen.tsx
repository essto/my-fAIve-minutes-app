import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ActivityIndicator,
  ScrollView,
  TextInput,
  TouchableOpacity,
} from 'react-native';
import { useDietData, Meal } from '../hooks/useDietData';

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

    // For MVP we just log the name and a dummy product as actual scanning is not integrated yet.
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
    return (
      <View style={[styles.container, styles.center]}>
        <ActivityIndicator testID="diet-loading" size="large" color="#007AFF" />
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Twoja Dieta</Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Podsumowanie Dnia</Text>
        {dailySummary && (
          <View>
            <Text style={styles.summaryText}>Kalorie: {dailySummary.total.calories} kcal</Text>
            <View style={styles.macrosRow}>
              <Text style={styles.macroText}>Białko: {dailySummary.total.protein}g</Text>
              <Text style={styles.macroText}>Węg.: {dailySummary.total.carbs}g</Text>
              <Text style={styles.macroText}>Tł.: {dailySummary.total.fat}g</Text>
            </View>
          </View>
        )}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Dodaj Posiłek</Text>
        <TextInput
          style={styles.input}
          placeholder="Nazwa posiłku (np. Obiad)"
          value={newMealName}
          onChangeText={setNewMealName}
        />
        <TouchableOpacity style={styles.button} onPress={handleAddMeal} disabled={isLoading}>
          <Text style={styles.buttonText}>Dodaj posiłek</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Dzisiejsze Posiłki</Text>
        {meals.length === 0 ? (
          <Text style={styles.emptyText}>Brak posiłków</Text>
        ) : (
          meals.map((meal: Meal) => (
            <View key={meal.id} style={styles.mealItem}>
              <Text style={styles.mealName}>{meal.name}</Text>
            </View>
          ))
        )}
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
  summaryText: {
    fontSize: 22,
    fontWeight: '700',
    color: '#007AFF',
    marginBottom: 10,
  },
  macrosRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  macroText: {
    fontSize: 14,
    color: '#666',
  },
  input: {
    backgroundColor: '#FAF9F6',
    padding: 15,
    borderRadius: 10,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#EAEAEA',
    fontSize: 16,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  emptyText: {
    color: '#999',
    fontStyle: 'italic',
  },
  mealItem: {
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#F0F0F0',
  },
  mealName: {
    fontSize: 16,
    color: '#333',
  },
});
