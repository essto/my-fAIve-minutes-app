module.exports = {
  preset: 'react-native',
  setupFiles: ['<rootDir>/jest-setup.js'],
  setupFilesAfterEnv: ['@testing-library/jest-native/extend-expect'],
  moduleNameMapper: {
    '^react$': require.resolve('react'),
    '^react-test-renderer$': require.resolve('react-test-renderer'),
    '\\.css$': '<rootDir>/__mocks__/styleMock.js',
    'expo-blur': '<rootDir>/__mocks__/expo-blur.js',
  },
  transformIgnorePatterns: [
    'node_modules/(?!((jest-)?react-native|@react-native(-community)?)|expo(nent)?|@expo(nent)?/.*|@expo-google-fonts/.*|react-navigation|@react-navigation/.*|react-native-reanimated|@unimodules/.*|unimodules|sentry-expo|native-base|react-native-svg)',
  ],
};
