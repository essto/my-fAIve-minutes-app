import mockAsyncStorage from '@react-native-async-storage/async-storage/jest/async-storage-mock';

jest.mock('@react-native-async-storage/async-storage', () => mockAsyncStorage);

// Bypass Expo internal require error inside Jest
global.__ExpoImportMetaRegistry = {
  get: jest.fn(),
  set: jest.fn(),
};
