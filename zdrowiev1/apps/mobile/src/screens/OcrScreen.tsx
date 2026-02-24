import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
} from 'react-native';
import { useOCR } from '../hooks/useOCR';

export const OcrScreen = () => {
  const { ocrResult, isLoading, error, takePhoto, pickFromGallery, clearResult } = useOCR();

  if (isLoading) {
    return (
      <View style={[styles.container, styles.center]}>
        <ActivityIndicator testID="ocr-loading" size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Przetwarzanie obrazu...</Text>
      </View>
    );
  }

  if (ocrResult) {
    return (
      <ScrollView style={styles.container} contentContainerStyle={styles.centerTop}>
        <View style={styles.resultCard}>
          <Text style={styles.resultTitle}>Wynik rozpoznawania:</Text>
          <Text style={styles.resultText}>{ocrResult.text}</Text>
          <Text style={styles.confidenceText}>
            Pewność: {(ocrResult.confidence * 100).toFixed(0)}%
          </Text>
        </View>

        <TouchableOpacity style={styles.buttonSecondary} onPress={clearResult}>
          <Text style={styles.buttonTextSecondary}>Skanuj ponownie</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.buttonPrimary} onPress={() => {}}>
          <Text style={styles.buttonTextPrimary}>Zapisz jako wynik badań</Text>
        </TouchableOpacity>
      </ScrollView>
    );
  }

  return (
    <View style={[styles.container, styles.center]}>
      <Text style={styles.title}>Analiza Badań i Etykiet</Text>
      <Text style={styles.subtitle}>
        Zeskanuj zdjęcie swoich wyników badań lub etykiety produktu spożywczego.
      </Text>

      {error && <Text style={styles.errorText}>{error}</Text>}

      <TouchableOpacity style={styles.buttonPrimary} onPress={takePhoto}>
        <Text style={styles.buttonTextPrimary}>Zrób zdjęcie</Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.buttonSecondary} onPress={pickFromGallery}>
        <Text style={styles.buttonTextSecondary}>Wybierz z galerii</Text>
      </TouchableOpacity>
    </View>
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
  centerTop: {
    alignItems: 'center',
    paddingTop: 40,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1A1A1A',
    marginBottom: 10,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 40,
    paddingHorizontal: 10,
  },
  errorText: {
    color: '#D32F2F',
    marginBottom: 20,
    textAlign: 'center',
  },
  loadingText: {
    marginTop: 20,
    fontSize: 16,
    color: '#666',
  },
  buttonPrimary: {
    backgroundColor: '#007AFF',
    paddingVertical: 15,
    paddingHorizontal: 40,
    borderRadius: 12,
    width: '100%',
    alignItems: 'center',
    marginBottom: 15,
  },
  buttonTextPrimary: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  buttonSecondary: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#007AFF',
    paddingVertical: 15,
    paddingHorizontal: 40,
    borderRadius: 12,
    width: '100%',
    alignItems: 'center',
    marginBottom: 15,
  },
  buttonTextSecondary: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: '600',
  },
  resultCard: {
    backgroundColor: '#fff',
    width: '100%',
    padding: 20,
    borderRadius: 15,
    marginBottom: 30,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 5,
    elevation: 2,
  },
  resultTitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 10,
  },
  resultText: {
    fontSize: 18,
    color: '#1A1A1A',
    fontWeight: '500',
    marginBottom: 15,
  },
  confidenceText: {
    fontSize: 12,
    color: '#4CAF50',
    textAlign: 'right',
  },
});
