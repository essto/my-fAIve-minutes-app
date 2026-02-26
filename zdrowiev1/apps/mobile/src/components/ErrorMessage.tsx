import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';

interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({ message, onRetry }) => {
  return (
    <View className="flex-1 bg-background justify-center items-center p-6">
      <Text className="text-destructive font-medium mb-4 text-center">{message}</Text>
      {onRetry && (
        <TouchableOpacity
          className="border border-destructive py-3 px-8 rounded-xl items-center active:bg-destructive/10 cursor-pointer"
          onPress={onRetry}
        >
          <Text className="text-destructive font-bold text-base">Ponów</Text>
        </TouchableOpacity>
      )}
    </View>
  );
};
