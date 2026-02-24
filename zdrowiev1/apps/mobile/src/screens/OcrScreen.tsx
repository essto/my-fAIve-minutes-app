import React from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, ScrollView } from 'react-native';
import { useOCR } from '../hooks/useOCR';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { BlurView } from 'expo-blur';

export const OcrScreen = () => {
  const { ocrResult, isLoading, error, takePhoto, pickFromGallery, clearResult } = useOCR();

  if (isLoading) {
    return (
      <View className="flex-1 bg-background justify-center items-center">
        <ActivityIndicator testID="ocr-loading" size="large" color="#8251EE" />
        <Text className="text-muted-foreground mt-6 text-base font-medium">
          Przetwarzanie obrazu...
        </Text>
      </View>
    );
  }

  if (ocrResult) {
    return (
      <ScrollView className="flex-1 bg-background p-6">
        <Animated.View
          entering={FadeInDown.springify()}
          className="mt-10 mb-8 overflow-hidden rounded-3xl border border-border"
        >
          <BlurView intensity={30} tint="dark" className="p-8">
            <Text className="text-secondary-foreground font-medium mb-3 opacity-80 uppercase tracking-widest text-xs">
              Wynik Rozpoznawania
            </Text>
            <Text className="text-xl text-foreground font-medium mb-6 leading-relaxed bg-neutral-bg3 p-4 rounded-xl border border-border/50">
              {ocrResult.text}
            </Text>
            <Text className="text-brand font-bold text-sm text-right">
              Pewność: {(ocrResult.confidence * 100).toFixed(0)}%
            </Text>
          </BlurView>
        </Animated.View>

        <Animated.View entering={FadeInUp.delay(100).springify()}>
          <TouchableOpacity
            className="border border-brand py-4 rounded-xl items-center flex-row justify-center active:bg-brand/10 mb-4"
            onPress={clearResult}
          >
            <Text className="text-brand font-bold">Skanuj ponownie</Text>
          </TouchableOpacity>
        </Animated.View>

        <Animated.View entering={FadeInUp.delay(200).springify()}>
          <TouchableOpacity
            className="bg-brand py-4 rounded-xl items-center flex-row justify-center active:bg-brand-hover mb-8"
            onPress={() => {}}
          >
            <Text className="text-white font-bold">Zapisz jako wynik badań</Text>
          </TouchableOpacity>
        </Animated.View>
      </ScrollView>
    );
  }

  return (
    <View className="flex-1 bg-background justify-center p-6">
      <Animated.Text
        entering={FadeInDown.springify()}
        className="text-3xl font-bold text-foreground text-center mb-4"
      >
        Skaner Etykiet i Badań
      </Animated.Text>
      <Animated.Text
        entering={FadeInDown.delay(100).springify()}
        className="text-muted-foreground text-center mb-10 text-base px-2"
      >
        Precyzyjne rozpoznawanie tekstu napędzane przez model Google Vision. Zrób zdjęcie aby
        wyodrębnić wartości.
      </Animated.Text>

      {error ? (
        <Text className="text-destructive text-center font-medium mb-6">{error}</Text>
      ) : null}

      <Animated.View entering={FadeInUp.delay(200).springify()}>
        <TouchableOpacity
          className="bg-brand py-4 rounded-xl items-center flex-row justify-center active:bg-brand-hover mb-4"
          onPress={takePhoto}
        >
          <Text className="text-white font-bold text-base">Uruchom Aparat</Text>
        </TouchableOpacity>
      </Animated.View>

      <Animated.View entering={FadeInUp.delay(300).springify()}>
        <TouchableOpacity
          className="border border-brand py-4 rounded-xl items-center flex-row justify-center active:bg-brand/10"
          onPress={pickFromGallery}
        >
          <Text className="text-brand font-bold text-base">Wybierz z Galerii</Text>
        </TouchableOpacity>
      </Animated.View>
    </View>
  );
};
