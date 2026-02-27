import React, { useEffect } from 'react';
import { View, Text } from 'react-native';
import Svg, { Circle, G } from 'react-native-svg';
import Animated, {
  useAnimatedProps,
  useSharedValue,
  withSpring,
  withDelay,
} from 'react-native-reanimated';

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

interface ActivityRingsProps {
  stepsProgress: number; // 0 to 1
  caloriesProgress: number;
  exerciseProgress: number;
  size?: number;
}

export const ActivityRings = ({
  stepsProgress,
  caloriesProgress,
  exerciseProgress,
  size = 140,
}: ActivityRingsProps) => {
  const strokeWidth = 12;
  const gap = 2; // gap between rings

  const center = size / 2;
  const r1 = (size - strokeWidth) / 2;
  const r2 = r1 - strokeWidth - gap;
  const r3 = r2 - strokeWidth - gap;

  // Animations
  const p1 = useSharedValue(0);
  const p2 = useSharedValue(0);
  const p3 = useSharedValue(0);

  useEffect(() => {
    p1.value = withSpring(Math.min(stepsProgress, 1), { damping: 15 });
    p2.value = withDelay(100, withSpring(Math.min(caloriesProgress, 1), { damping: 15 }));
    p3.value = withDelay(200, withSpring(Math.min(exerciseProgress, 1), { damping: 15 }));
  }, [stepsProgress, caloriesProgress, exerciseProgress]);

  const makeAnimatedProps = (radius: number, progressObj: Animated.SharedValue<number>) => {
    const circumference = radius * 2 * Math.PI;
    return useAnimatedProps(() => ({
      strokeDashoffset: circumference - circumference * progressObj.value,
      strokeDasharray: `${circumference} ${circumference}`,
    }));
  };

  const props1 = makeAnimatedProps(r1, p1);
  const props2 = makeAnimatedProps(r2, p2);
  const props3 = makeAnimatedProps(r3, p3);

  return (
    <View className="items-center justify-center relative" style={{ width: size, height: size }}>
      <Svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <G rotation="-90" originX={center} originY={center}>
          {/* Backgrounds */}
          <Circle
            cx={center}
            cy={center}
            r={r1}
            stroke="#E11D4833"
            strokeWidth={strokeWidth}
            fill="none"
          />
          <Circle
            cx={center}
            cy={center}
            r={r2}
            stroke="#10B98133"
            strokeWidth={strokeWidth}
            fill="none"
          />
          <Circle
            cx={center}
            cy={center}
            r={r3}
            stroke="#0EA5E933"
            strokeWidth={strokeWidth}
            fill="none"
          />

          {/* Foreground Rings */}
          <AnimatedCircle
            testID="ring-steps"
            cx={center}
            cy={center}
            r={r1}
            stroke="#E11D48"
            strokeWidth={strokeWidth}
            fill="none"
            strokeLinecap="round"
            animatedProps={props1}
          />
          <AnimatedCircle
            testID="ring-calories"
            cx={center}
            cy={center}
            r={r2}
            stroke="#10B981"
            strokeWidth={strokeWidth}
            fill="none"
            strokeLinecap="round"
            animatedProps={props2}
          />
          <AnimatedCircle
            testID="ring-exercise"
            cx={center}
            cy={center}
            r={r3}
            stroke="#0EA5E9"
            strokeWidth={strokeWidth}
            fill="none"
            strokeLinecap="round"
            animatedProps={props3}
          />
        </G>
      </Svg>
      <View className="absolute items-center justify-center">
        <Text className="text-2xl">🏃</Text>
      </View>
    </View>
  );
};
