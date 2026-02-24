import mockAsyncStorage from '@react-native-async-storage/async-storage/jest/async-storage-mock';
jest.mock('@react-native-async-storage/async-storage', () => mockAsyncStorage);

import AsyncStorage from '@react-native-async-storage/async-storage';
import { storage } from '../storage';

describe('StorageService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('T5.1: should save and retrieve auth token', async () => {
    await storage.setToken('test-jwt-123');
    expect(AsyncStorage.setItem).toHaveBeenCalledWith('auth_token', 'test-jwt-123');

    (AsyncStorage.getItem as jest.Mock).mockResolvedValue('test-jwt-123');
    const token = await storage.getToken();
    expect(token).toBe('test-jwt-123');
  });

  it('T5.2: should remove token on clear', async () => {
    await storage.removeToken();
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith('auth_token');
  });

  it('T5.3: should return null when no token stored', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
    const token = await storage.getToken();
    expect(token).toBeNull();
  });
});
