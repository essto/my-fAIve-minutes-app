const React = require('react');
const { View, Text, Image, ScrollView, Animated } = require('react-native');

const animationBuilder = () => {
  const self = {
    delay: () => self,
    springify: () => self,
    duration: () => self,
    damping: () => self,
    mass: () => self,
    stiffness: () => self,
    withCallback: () => self,
    randomDelay: () => self,
    type: () => self,
    build: () => ({}),
  };
  return self;
};

const Reanimated = {
  ...Animated,
  View: View,
  Text: Text,
  Image: Image,
  ScrollView: ScrollView,
  createAnimatedComponent: (v) => v,
  addWhitelistedNativeProps: () => {},
  addWhitelistedUIProps: () => {},

  // animations
  FadeIn: animationBuilder(),
  FadeInDown: animationBuilder(),
  FadeInUp: animationBuilder(),
  FadeInLeft: animationBuilder(),
  FadeInRight: animationBuilder(),
  FadeOut: animationBuilder(),
  Layout: animationBuilder(),
  SlideInLeft: animationBuilder(),
  SlideInRight: animationBuilder(),

  // mocks
  useSharedValue: (v) => ({ value: v }),
  useAnimatedStyle: (cb) => cb() || {},
  withTiming: (v) => v,
  withSpring: (v) => v,
  withRepeat: (v) => v,
  withSequence: (v) => v,
  runOnJS: (fn) => fn,
  useAnimatedReaction: () => {},
  useAnimatedRef: () => ({ current: null }),
  useDerivedValue: (cb) => ({ value: cb() }),
  interpolate: (v, input, output) => v,
  Extrapolation: { CLAMP: 'clamp' },
};

module.exports = Reanimated;
