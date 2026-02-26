import React from 'react';
import { View, Text, TouchableOpacity, ScrollView } from 'react-native';
import { useAuth } from '../hooks/useAuth';
import { useProfileStore } from '../store/useProfileStore';
import { FadeInUp, Layout } from 'react-native-reanimated';
import { GlassCard } from '../components/GlassCard';
import { ScreenHeader } from '../components/ScreenHeader';

export const ProfileScreen = () => {
  const { user, logout } = useAuth();

  // Zustand selectors (good practice to pick only what's needed)
  const language = useProfileStore((state) => state.language);
  const theme = useProfileStore((state) => state.theme);
  const setLanguage = useProfileStore((state) => state.setLanguage);
  const setTheme = useProfileStore((state) => state.setTheme);

  return (
    <ScrollView className="flex-1 bg-background p-6">
      <ScreenHeader title="Mój Profil" />

      <GlassCard entering={FadeInUp.springify()} className="mb-6 p-6 items-center">
        <View className="w-24 h-24 rounded-full bg-brand/20 items-center justify-center mb-4 border border-brand/50">
          <Text className="text-brand text-4xl font-bold">
            {user?.email?.charAt(0).toUpperCase() || 'U'}
          </Text>
        </View>
        <Text className="text-foreground text-2xl font-bold text-center">
          {/* user.name is technically not in the interface initially for the prompt but for the sake of completeness */}
          {(user as any)?.name || 'Użytkownik'}
        </Text>
        <Text className="text-muted-foreground text-base text-center mt-1">
          {user?.email || 'Brak danych'}
        </Text>
      </GlassCard>

      <Text className="text-muted-foreground font-semibold text-sm mb-2 ml-2 uppercase tracking-widest">
        Ustawienia
      </Text>

      <GlassCard entering={FadeInUp.delay(100).springify()} className="mb-6 p-2 px-4">
        <TouchableOpacity
          className="py-4 border-b border-border/50 flex-row justify-between items-center"
          onPress={() => setLanguage(language === 'pl' ? 'en' : 'pl')}
        >
          <Text className="text-foreground text-lg">Zmień język (PL/EN)</Text>
          <Text className="text-brand font-medium text-base uppercase">{language}</Text>
        </TouchableOpacity>

        <TouchableOpacity
          className="py-4 border-b border-border/50 flex-row justify-between items-center"
          onPress={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        >
          <Text className="text-foreground text-lg">Zmień motyw</Text>
          <Text className="text-brand font-medium text-base capitalize">{theme}</Text>
        </TouchableOpacity>

        <TouchableOpacity className="py-4 flex-row justify-between items-center">
          <Text className="text-foreground text-lg">Zgody i prywatność</Text>
          <Text className="text-muted-foreground font-medium text-lg">{'>'}</Text>
        </TouchableOpacity>
      </GlassCard>

      <GlassCard entering={FadeInUp.delay(200).springify()} intensity={10} className="mb-10 p-2">
        <TouchableOpacity className="py-4 w-full items-center" onPress={logout}>
          <Text className="text-destructive font-bold text-lg">Wyloguj się</Text>
        </TouchableOpacity>
      </GlassCard>
    </ScrollView>
  );
};
