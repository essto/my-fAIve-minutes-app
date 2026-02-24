import { renderHook, act, waitFor } from '@testing-library/react-native';
import { useAuth } from '../useAuth';
import api from '../../services/api';
import { storage } from '../../services/storage';

jest.mock('../../services/api');
jest.mock('../../services/storage');

const mockApi = api as jest.Mocked<typeof api>;
const mockStorage = storage as jest.Mocked<typeof storage>;

describe('useAuth', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('T1.1: should start with isAuthenticated=false and isLoading=true', async () => {
    mockStorage.getToken.mockResolvedValue(null);
    const { result } = renderHook(() => useAuth());
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.isLoading).toBe(true);
    expect(result.current.user).toBeNull();
    await waitFor(() => expect(result.current.isLoading).toBe(false));
  });

  it('T1.2: should restore session when valid token exists in storage', async () => {
    mockStorage.getToken.mockResolvedValue('valid-jwt-token');
    mockApi.get.mockResolvedValue({ data: { id: 'u1', email: 'test@zdrowie.pl' } });

    const { result } = renderHook(() => useAuth());

    await waitFor(() => {
      expect(result.current.isAuthenticated).toBe(true);
      expect(result.current.user?.email).toBe('test@zdrowie.pl');
      expect(result.current.isLoading).toBe(false);
    });
  });

  it('T1.3: should set isAuthenticated=false when stored token is expired/invalid', async () => {
    mockStorage.getToken.mockResolvedValue('expired-token');
    mockApi.get.mockRejectedValue({ response: { status: 401 } });

    const { result } = renderHook(() => useAuth());

    await waitFor(() => {
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.isLoading).toBe(false);
    });
    expect(mockStorage.removeToken).toHaveBeenCalled();
  });

  it('T1.4: should login successfully and save token', async () => {
    mockApi.post.mockResolvedValue({
      data: { access_token: 'new-jwt', user: { id: 'u1', email: 'jan@test.pl' } },
    });

    // Default mock setup for restore session
    mockStorage.getToken.mockResolvedValue(null);

    const { result } = renderHook(() => useAuth());

    // wait for initial restore to finish
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.login('jan@test.pl', 'haslo123');
    });

    expect(mockStorage.setToken).toHaveBeenCalledWith('new-jwt');
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user?.email).toBe('jan@test.pl');
  });

  it('T1.5: should set error on invalid credentials (401)', async () => {
    mockStorage.getToken.mockResolvedValue(null);
    mockApi.post.mockRejectedValue({
      response: { status: 401, data: { message: 'Invalid credentials' } },
    });

    const { result } = renderHook(() => useAuth());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.login('bad@test.pl', 'wrong');
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.error).toBe('Invalid credentials');
  });

  it('T1.6: should handle network timeout gracefully', async () => {
    mockStorage.getToken.mockResolvedValue(null);
    mockApi.post.mockRejectedValue(new Error('Network Error'));

    const { result } = renderHook(() => useAuth());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.login('jan@test.pl', 'haslo123');
    });

    expect(result.current.error).toContain('Network');
    expect(result.current.isAuthenticated).toBe(false);
  });

  it('T1.7: should clear token and reset state on logout', async () => {
    mockStorage.getToken.mockResolvedValue('valid-token');
    mockApi.get.mockResolvedValue({ data: { id: 'u1', email: 'test@zdrowie.pl' } });

    const { result } = renderHook(() => useAuth());
    await waitFor(() => expect(result.current.isAuthenticated).toBe(true));

    await act(async () => {
      await result.current.logout();
    });

    expect(mockStorage.removeToken).toHaveBeenCalled();
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });
});
