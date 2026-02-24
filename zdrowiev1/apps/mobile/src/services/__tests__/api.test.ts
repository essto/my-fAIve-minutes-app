import mockAsyncStorage from '@react-native-async-storage/async-storage/jest/async-storage-mock';
jest.mock('@react-native-async-storage/async-storage', () => mockAsyncStorage);

import axios from 'axios';
import api from '../api';
import { storage } from '../storage';

jest.mock('../storage');

describe('Mobile API Client', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('T6.1: should include auth token in request headers', async () => {
    const mockToken = 'jwt-token';
    (storage.getToken as jest.Mock).mockResolvedValue(mockToken);

    // We need to mock the underlying axios adapter or mock axios.get directly
    // Wait, `api` is an axios instance. We can intercept its request, or check its interceptor.
    // A simpler way: mock axios standard adapter to return a 200, and we capture the config.
    const mockAdapter = jest.fn().mockResolvedValue({ data: 'ok', status: 200 });
    api.defaults.adapter = mockAdapter;

    await api.get('/health/weight');

    expect(mockAdapter).toHaveBeenCalled();
    const config = mockAdapter.mock.calls[0][0];
    expect(config.headers['Authorization']).toBe(`Bearer ${mockToken}`);
  });

  it('T6.2: should use baseURL from env config (API_URL)', () => {
    // defaults to http://localhost:3001/api if EXPO_PUBLIC_API_URL is not set
    expect(api.defaults.baseURL).toContain('/api');
  });

  it('T6.3: should handle 401 by clearing token', async () => {
    const mockAdapter = jest.fn().mockRejectedValue({ response: { status: 401 } });
    api.defaults.adapter = mockAdapter;

    await expect(api.get('/protected')).rejects.toBeTruthy();
    expect(storage.removeToken).toHaveBeenCalled();
  });

  it('T6.4: should have 10s timeout', () => {
    expect(api.defaults.timeout).toBe(10000);
  });
});
