import React from 'react';
import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { View, ActivityIndicator } from 'react-native';

import { useAuth } from '../hooks/useAuth';
import { LoginScreen } from '../screens/LoginScreen';
import { OcrScreen } from '../screens/OcrScreen';
import { ScaleScreen } from '../screens/ScaleScreen';
import { HomeScreen } from '../screens/HomeScreen';
import { HealthScreen } from '../screens/HealthScreen';
import { DietScreen } from '../screens/DietScreen';
import { ProfileScreen } from '../screens/ProfileScreen';
import { WatchScreen } from '../screens/WatchScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

const AppDarkTheme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    background: '#1d1d1f',
    card: '#18181A',
    text: '#ffffff',
    border: '#2a2a2d',
    primary: '#8251EE',
  },
};

export const TabNavigator = () => (
  <Tab.Navigator
    screenOptions={{
      headerShown: true,
      headerStyle: { backgroundColor: '#18181A' },
      headerTintColor: '#fff',
      tabBarStyle: { backgroundColor: '#18181A', borderTopColor: '#2a2a2d' },
      tabBarActiveTintColor: '#8251EE',
      tabBarInactiveTintColor: '#666',
    }}
  >
    <Tab.Screen name="Home" component={HomeScreen} />
    <Tab.Screen name="Health" component={HealthScreen} />
    <Tab.Screen name="Diet" component={DietScreen} />
    <Tab.Screen name="Skaner" component={OcrScreen} />
    <Tab.Screen name="Waga" component={ScaleScreen} />
    <Tab.Screen name="Profil" component={ProfileScreen} />
  </Tab.Navigator>
);

export const AuthStack = () => (
  <Stack.Navigator>
    <Stack.Screen name="Login" component={LoginScreen} options={{ headerShown: false }} />
  </Stack.Navigator>
);

export const MainStack = () => (
  <Stack.Navigator>
    <Stack.Screen name="Tabs" component={TabNavigator} options={{ headerShown: false }} />
    <Stack.Screen
      name="WatchScreen"
      component={WatchScreen}
      options={{ title: 'Zegarki i Opaski' }}
    />
  </Stack.Navigator>
);

export const RootNavigator = () => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <View
        style={{
          flex: 1,
          backgroundColor: '#1d1d1f',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <ActivityIndicator testID="root-loading" size="large" color="#8251EE" />
      </View>
    );
  }

  return (
    <NavigationContainer theme={AppDarkTheme}>
      {isAuthenticated ? <MainStack /> : <AuthStack />}
    </NavigationContainer>
  );
};
