import mockAsyncStorage from '@react-native-async-storage/async-storage/jest/async-storage-mock';

jest.mock('@react-native-async-storage/async-storage', () => mockAsyncStorage);

// Bypass Expo internal require error inside Jest
global.__ExpoImportMetaRegistry = {
  get: jest.fn(),
  set: jest.fn(),
};

// Mock NativeWind's css-interop to prevent displayName crashes while maintaining babel transform stability
jest.mock('react-native-css-interop', () => ({
  cssInterop: jest.fn(),
  remapProps: jest.fn(),
  createInteropElement: require('react').createElement,
  createElement: require('react').createElement,
}));

jest.mock('react-native-reanimated', () => {
  const React = require('react');
  const { View, Text } = require('react-native');
  return {
    __esModule: true,
    default: {
      View: View,
      Text: Text,
      createAnimatedComponent: (Component) => Component,
    },
    useSharedValue: jest.fn(() => ({ value: 0 })),
    useAnimatedStyle: jest.fn(() => ({})),
    useAnimatedProps: jest.fn(() => ({})),
    withTiming: jest.fn((v) => v),
    withSpring: jest.fn((v) => v),
    withDelay: jest.fn((d, v) => v),
    withSequence: jest.fn((...args) => args[0]),
    FadeInUp: {
      delay: jest.fn().mockReturnThis(),
      springify: jest.fn().mockReturnThis(),
    },
    FadeInDown: {
      delay: jest.fn().mockReturnThis(),
      springify: jest.fn().mockReturnThis(),
    },
    Layout: {
      springify: jest.fn().mockReturnThis(),
    },
    createAnimatedComponent: (Component) => Component,
    Animated: {
      View: View,
      Text: Text,
      createAnimatedComponent: (Component) => Component,
    },
  };
});

jest.mock('expo-image-picker', () => ({
  launchCameraAsync: jest.fn(),
  launchImageLibraryAsync: jest.fn(),
  requestCameraPermissionsAsync: jest.fn(),
  requestMediaLibraryPermissionsAsync: jest.fn(),
  PermissionStatus: {
    GRANTED: 'granted',
    DENIED: 'denied',
    UNDETERMINED: 'undetermined',
  },
  MediaTypeOptions: {
    Images: 'Images',
    Videos: 'Videos',
    All: 'All',
  },
}));

jest.mock('expo-camera', () => {
  const React = require('react');
  const View = require('react-native').View;

  return {
    useCameraPermissions: () => [{ granted: true, status: 'granted' }, jest.fn()],
    CameraView: ({ onBarcodeScanned, testID, children }) => {
      return (
        <View
          testID={testID}
          onLayout={() => onBarcodeScanned && onBarcodeScanned({ data: '5901234' })}
        >
          {children}
        </View>
      );
    },
  };
});
