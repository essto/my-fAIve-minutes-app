import React from 'react';
import { View, ViewProps } from 'react-native';
import { BlurView, BlurViewProps } from 'expo-blur';
import Animated, { AnimatedProps } from 'react-native-reanimated';

export interface GlassCardProps extends AnimatedProps<ViewProps> {
  intensity?: number;
  tint?: BlurViewProps['tint'];
  children: React.ReactNode;
}

export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  intensity = 30,
  tint = 'dark',
  className,
  ...rest
}) => {
  return (
    <Animated.View
      className={`mb-6 overflow-hidden rounded-3xl border border-border ${className}`}
      {...rest}
    >
      <BlurView intensity={intensity} tint={tint} className="p-6">
        {children}
      </BlurView>
    </Animated.View>
  );
};
