import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useAuth } from '../hooks/useAuth';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { BlurView } from 'expo-blur';

export const LoginScreen = () => {
  const { login, isLoading, error } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = () => {
    if (email && password) {
      login(email, password);
    }
  };

  return (
    <View className="flex-1 justify-center p-6 bg-background">
      <Animated.Text
        entering={FadeInDown.springify()}
        className="text-4xl font-bold text-center mb-10 text-brand"
      >
        Zdrowie
      </Animated.Text>

      {error ? (
        <Animated.Text entering={FadeInDown} className="text-red-500 font-medium text-center mb-6">
          {error}
        </Animated.Text>
      ) : null}

      <Animated.View
        entering={FadeInUp.delay(100).springify()}
        className="mb-4 overflow-hidden rounded-2xl border border-border"
      >
        <BlurView intensity={20} tint="dark" className="p-4">
          <TextInput
            className="text-foreground text-base py-2"
            placeholder="Email"
            placeholderTextColor="#666"
            value={email}
            onChangeText={setEmail}
            autoCapitalize="none"
            keyboardType="email-address"
          />
        </BlurView>
      </Animated.View>

      <Animated.View
        entering={FadeInUp.delay(200).springify()}
        className="mb-8 overflow-hidden rounded-2xl border border-border"
      >
        <BlurView intensity={20} tint="dark" className="p-4">
          <TextInput
            className="text-foreground text-base py-2"
            placeholder="Hasło"
            placeholderTextColor="#666"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
          />
        </BlurView>
      </Animated.View>

      <Animated.View entering={FadeInUp.delay(300).springify()}>
        <TouchableOpacity
          className="bg-brand py-4 rounded-2xl items-center flex-row justify-center active:bg-brand-hover min-h-[56px]"
          onPress={handleLogin}
          disabled={isLoading}
        >
          {isLoading ? (
            <ActivityIndicator testID="loading-indicator" color="#fff" />
          ) : (
            <Text className="text-white text-lg font-bold">Zaloguj</Text>
          )}
        </TouchableOpacity>
      </Animated.View>
    </View>
  );
};
