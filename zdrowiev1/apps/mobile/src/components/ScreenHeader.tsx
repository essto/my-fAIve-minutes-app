import React from 'react';
import Animated, { FadeInDown } from 'react-native-reanimated';

interface ScreenHeaderProps {
  title: string;
}

export const ScreenHeader: React.FC<ScreenHeaderProps> = ({ title }) => {
  return (
    <Animated.Text
      entering={FadeInDown.springify()}
      className="text-3xl font-bold text-foreground mb-8 mt-2"
    >
      {title}
    </Animated.Text>
  );
};
