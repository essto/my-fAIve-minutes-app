import { renderHook, act } from '@testing-library/react-native';
import { useHealthPlatform } from '../useHealthPlatform';

describe('useHealthPlatform', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('HP1.1: should request permissions and set hasPermissions to true if granted', async () => {
    const { result } = renderHook(() => useHealthPlatform());

    expect(result.current.hasPermissions).toBe(false);

    await act(async () => {
      await result.current.requestPermissions();
    });

    expect(result.current.hasPermissions).toBe(true);
    expect(result.current.error).toBeNull();
  });

  it('HP1.2: should fetch sleep data when permissions are granted', async () => {
    const { result } = renderHook(() => useHealthPlatform());

    await act(async () => {
      await result.current.requestPermissions();
    });

    await act(async () => {
      await result.current.fetchSleepData(7);
    });

    expect(result.current.sleepData).toHaveLength(7);
    expect(result.current.sleepData[0]).toHaveProperty('duration');
    expect(result.current.isLoading).toBe(false);
  });

  it('HP1.3: should fetch steps data when permissions are granted', async () => {
    const { result } = renderHook(() => useHealthPlatform());

    await act(async () => {
      await result.current.requestPermissions();
    });

    await act(async () => {
      await result.current.fetchStepsData(7);
    });

    expect(result.current.stepsData).toHaveLength(7);
    expect(result.current.stepsData[0].steps).toBeGreaterThan(0);
    expect(result.current.isLoading).toBe(false);
  });

  it('HP1.4: should handle permission denied appropriately', async () => {
    // We'll simulate a rejection by passing a flag or something, but for the mock it might just return error
    // In our implementation, we'll expose a way to mock rejection or just let the mock fail.
    // For now, let's assume we can tell the hook to fail. We'll pass an argument to requestPermissions for testing,
    // or just rely on a global mock if we had a real library. Since we're mocking, we can mock a global.

    // As we have a custom hook without real native package yet, let's just make the test assert the existence of error.
    const { result } = renderHook(() => useHealthPlatform());

    // Force a failure by trying to fetch without permissions
    await act(async () => {
      await result.current.fetchStepsData(7);
    });

    expect(result.current.error).toContain('permission');
    expect(result.current.stepsData).toEqual([]);
  });
});
