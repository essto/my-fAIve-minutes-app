import React from 'react';
import { View, Text, TouchableOpacity, ScrollView } from 'react-native';
import { useOCR } from '../hooks/useOCR';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { GlassCard } from '../components/GlassCard';
import { ScreenHeader } from '../components/ScreenHeader';
import { LoadingScreen } from '../components/LoadingScreen';

export const OcrScreen = () => {
  const { ocrResult, isLoading, error, takePhoto, pickFromGallery, clearResult } = useOCR();

  if (isLoading) {
    return <LoadingScreen testID="ocr-loading" />;
  }

  if (ocrResult) {
    return (
      <ScrollView className="flex-1 bg-background p-6">
        <ScreenHeader title="Wynik SKANU" />

        <GlassCard entering={FadeInDown.springify()} intensity={30} className="p-2">
          <Text className="text-secondary-foreground font-medium mb-3 opacity-80 uppercase tracking-widest text-xs">
            Rozpoznany Tekst
          </Text>
          <View className="bg-black/20 p-4 rounded-2xl mb-4 border border-white/5">
            <Text className="text-lg text-foreground font-medium leading-relaxed">
              {ocrResult.text}
            </Text>
          </View>
          <Text className="text-brand font-bold text-sm text-right">
            Pewność: {(ocrResult.confidence * 100).toFixed(0)}%
          </Text>
        </GlassCard>

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
            <Text className="text-white font-bold">Zapisz wynik</Text>
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
        Skaner AI
      </Animated.Text>
      <Animated.Text
        entering={FadeInDown.delay(100).springify()}
        className="text-muted-foreground text-center mb-10 text-base px-2"
      >
        Zrób zdjęcie etykiety lub wyniku badań, aby AI automatycznie wyodrębniło kluczowe dane.
      </Animated.Text>

      {error ? (
        <Text testID="ocr-error" className="text-destructive text-center font-medium mb-6">
          {error}
        </Text>
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
