import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { View, Text, ActivityIndicator } from 'react-native';

import { useAuth } from '../hooks/useAuth';
import { LoginScreen } from '../screens/LoginScreen';
import { OcrScreen } from '../screens/OcrScreen';
import { ScaleScreen } from '../screens/ScaleScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

// Placeholder screens for Dashboard
const HomeScreen = () => (
  <View>
    <Text>Home</Text>
  </View>
);
const HealthScreen = () => (
  <View>
    <Text>Health</Text>
  </View>
);
const DietScreen = () => (
  <View>
    <Text>Diet</Text>
  </View>
);

export const TabNavigator = () => (
  <Tab.Navigator screenOptions={{ headerShown: true }}>
    <Tab.Screen name="Home" component={HomeScreen} />
    <Tab.Screen name="Health" component={HealthScreen} />
    <Tab.Screen name="Diet" component={DietScreen} />
    <Tab.Screen name="Skaner" component={OcrScreen} />
    <Tab.Screen name="Waga" component={ScaleScreen} />
  </Tab.Navigator>
);

export const AuthStack = () => (
  <Stack.Navigator>
    <Stack.Screen name="Login" component={LoginScreen} options={{ headerShown: false }} />
  </Stack.Navigator>
);

export const RootNavigator = () => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator testID="root-loading" size="large" color="#007AFF" />
      </View>
    );
  }

  return (
    <NavigationContainer>{isAuthenticated ? <TabNavigator /> : <AuthStack />}</NavigationContainer>
  );
};
