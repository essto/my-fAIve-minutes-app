import React, { useEffect } from 'react';
import { View, Text } from 'react-native';
import Svg, { Circle } from 'react-native-svg';
import Animated, {
  useAnimatedProps,
  useSharedValue,
  withTiming,
  withSpring,
} from 'react-native-reanimated';

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

interface HealthScoreRingProps {
  score: number;
  size?: number;
  strokeWidth?: number;
}

export const HealthScoreRing = ({ score, size = 120, strokeWidth = 12 }: HealthScoreRingProps) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const progress = useSharedValue(0);

  useEffect(() => {
    progress.value = withSpring(score / 100, {
      damping: 15,
      stiffness: 90,
    });
  }, [score]);

  const animatedProps = useAnimatedProps(() => {
    const strokeDashoffset = circumference - circumference * progress.value;
    return {
      strokeDashoffset,
    };
  });

  const getScoreColor = () => {
    if (score >= 80) return '#10B981'; // Green
    if (score >= 50) return '#F59E0B'; // Yellow/Amber
    return '#EF4444'; // Red
  };

  const finalColor = getScoreColor();

  return (
    <View className="items-center justify-center" style={{ width: size, height: size }}>
      <Svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background Ring */}
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#2a2a2d"
          strokeWidth={strokeWidth}
          fill="none"
        />
        {/* Foreground Ring */}
        <AnimatedCircle
          testID="health-ring-arc"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={finalColor}
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={circumference}
          animatedProps={animatedProps}
          strokeLinecap="round"
          rotation="-90"
          originX={size / 2}
          originY={size / 2}
        />
      </Svg>
      <View className="absolute items-center justify-center">
        <Text className="text-3xl font-bold text-foreground">{score}</Text>
        <Text className="text-xs text-muted-foreground uppercase tracking-wider">Wynik</Text>
      </View>
    </View>
  );
};
