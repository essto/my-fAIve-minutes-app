import React, { useEffect } from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withSequence,
  withTiming,
} from 'react-native-reanimated';

interface NotificationBellProps {
  count: number;
  onPress?: () => void;
}

export const NotificationBell = ({ count, onPress }: NotificationBellProps) => {
  const scale = useSharedValue(1);
  const rotation = useSharedValue(0);

  useEffect(() => {
    if (count > 0) {
      scale.value = withSequence(withSpring(1.2, { damping: 2 }), withSpring(1));
      rotation.value = withSequence(
        withTiming(-15, { duration: 100 }),
        withTiming(15, { duration: 100 }),
        withTiming(-15, { duration: 100 }),
        withTiming(15, { duration: 100 }),
        withTiming(0, { duration: 100 }),
      );
    }
  }, [count, scale, rotation]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }, { rotate: `${rotation.value}deg` }],
  }));

  return (
    <TouchableOpacity onPress={onPress} style={{ position: 'relative', padding: 8 }}>
      <Animated.View style={animatedStyle}>
        <Text style={{ fontSize: 24 }}>🔔</Text>
      </Animated.View>

      {count > 0 && (
        <View
          testID="notification-dot"
          style={{
            position: 'absolute',
            top: 4,
            right: 4,
            backgroundColor: '#ef4444',
            width: 20,
            height: 20,
            borderRadius: 10,
            alignItems: 'center',
            justifyContent: 'center',
            borderWidth: 2,
            borderColor: '#1d1d1f',
          }}
        >
          <Text style={{ fontSize: 10, color: '#fff', fontWeight: 'bold' }}>
            {count > 9 ? '9+' : count}
          </Text>
        </View>
      )}
    </TouchableOpacity>
  );
};
