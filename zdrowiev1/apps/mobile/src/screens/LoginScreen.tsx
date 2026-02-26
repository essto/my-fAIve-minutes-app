import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useAuth } from '../hooks/useAuth';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { GlassCard } from '../components/GlassCard';

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
        className="text-5xl font-bold text-center mb-10 text-brand"
        style={{ letterSpacing: -2 }}
      >
        Zdrowie
      </Animated.Text>

      {error ? (
        <Text testID="login-error" className="text-destructive font-medium text-center mb-6">
          {error}
        </Text>
      ) : null}

      <GlassCard entering={FadeInUp.delay(100).springify()} intensity={20} className="mb-4">
        <TextInput
          className="text-foreground text-base py-2"
          placeholder="Email"
          placeholderTextColor="#666"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
        />
      </GlassCard>

      <GlassCard entering={FadeInUp.delay(200).springify()} intensity={20} className="mb-8">
        <TextInput
          className="text-foreground text-base py-2"
          placeholder="Hasło"
          placeholderTextColor="#666"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />
      </GlassCard>

      <Animated.View entering={FadeInUp.delay(300).springify()}>
        <TouchableOpacity
          className="bg-brand py-4 rounded-2xl items-center flex-row justify-center active:bg-brand-hover min-h-[56px]"
          onPress={handleLogin}
          disabled={isLoading}
        >
          {isLoading ? (
            <ActivityIndicator testID="loading-indicator" color="#fff" />
          ) : (
            <Text className="text-white text-lg font-bold">Zaloguj się</Text>
          )}
        </TouchableOpacity>
      </Animated.View>

      <Animated.Text
        entering={FadeInDown.delay(500)}
        className="text-muted-foreground text-center mt-10 text-sm"
      >
        v1.0.0 • AI-Powered Health
      </Animated.Text>
    </View>
  );
};
