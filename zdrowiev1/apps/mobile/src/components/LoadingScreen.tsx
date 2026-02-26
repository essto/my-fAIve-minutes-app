import React from 'react';
import { View, ActivityIndicator } from 'react-native';

interface LoadingScreenProps {
  testID?: string;
  color?: string;
}

export const LoadingScreen: React.FC<LoadingScreenProps> = ({
  testID = 'loading',
  color = '#8251EE',
}) => {
  return (
    <View className="flex-1 bg-background justify-center items-center">
      <ActivityIndicator testID={testID} size="large" color={color} />
    </View>
  );
};
