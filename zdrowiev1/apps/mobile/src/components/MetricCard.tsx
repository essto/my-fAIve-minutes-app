import React from 'react';
import { View, Text } from 'react-native';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({ label, value, unit }) => {
  return (
    <View className="mb-2">
      <Text className="text-secondary-foreground font-medium mb-1 opacity-80 uppercase tracking-widest text-xs">
        {label}
      </Text>
      <View className="flex-row items-end gap-2">
        <Text className="text-4xl font-bold text-foreground">{value}</Text>
        <Text className="text-brand text-xl font-medium mb-1">{unit}</Text>
      </View>
    </View>
  );
};
