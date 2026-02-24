import { renderHook, act, waitFor } from '@testing-library/react-native';
import { useHealthData } from '../useHealthData';
import api from '../../services/api';

jest.mock('../../services/api');
const mockApi = api as jest.Mocked<typeof api>;

describe('useHealthData', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ----- T2.1 Pobranie historii wagi -----
  it('T2.1: should fetch weight history for last 30 days', async () => {
    mockApi.get.mockResolvedValue({
      data: [
        { id: 'w1', value: 82.5, timestamp: '2026-02-01T10:00:00Z', unit: 'kg' },
        { id: 'w2', value: 82.0, timestamp: '2026-02-08T10:00:00Z', unit: 'kg' },
      ],
    });

    const { result } = renderHook(() => useHealthData());

    await act(async () => {
      await result.current.fetchWeightHistory(30);
    });

    await waitFor(() => {
      expect(result.current.weightHistory).toHaveLength(2);
      expect(result.current.weightHistory[0].value).toBe(82.5);
    });
  });

  // ----- T2.2 Walidacja danych Zod -----
  it('T2.2: should expose validation errors when API returns invalid data', async () => {
    mockApi.get.mockResolvedValue({
      data: [{ id: 'w1', value: 999, timestamp: 'invalid-date' }], // Invalid data
    });

    const { result } = renderHook(() => useHealthData());

    await act(async () => {
      await result.current.fetchWeightHistory(30);
    });

    expect(result.current.validationErrors).toBeTruthy();
    expect(result.current.weightHistory).toHaveLength(0); // Should not load invalid data
  });

  // ----- T2.3 Health Score -----
  it('T2.3: should fetch health score (0-100)', async () => {
    mockApi.get.mockResolvedValue({
      data: { score: 85, breakdown: { weight: 90, sleep: 80, activity: 85 } },
    });

    const { result } = renderHook(() => useHealthData());

    await act(async () => {
      await result.current.fetchHealthScore();
    });

    await waitFor(() => {
      expect(result.current.healthScore).toBe(85);
      expect(result.current.healthBreakdown?.weight).toBe(90);
    });
  });

  // ----- T2.4 Anomalie -----
  it('T2.4: should fetch anomalies list', async () => {
    mockApi.get.mockResolvedValue({
      data: [{ id: 'a1', metric: 'Tętno', value: 120, severity: 'high', message: 'Wysokie tętno' }],
    });

    const { result } = renderHook(() => useHealthData());

    await act(async () => {
      await result.current.fetchAnomalies();
    });

    await waitFor(() => {
      expect(result.current.anomalies).toHaveLength(1);
      expect(result.current.anomalies[0].severity).toBe('high');
    });
  });

  // ----- T2.5 Stan ładowania -----
  it('T2.5: should set isLoading during fetch', async () => {
    let resolvePromise: (value: any) => void;
    mockApi.get.mockReturnValue(
      new Promise((res) => {
        resolvePromise = res;
      }),
    );

    const { result } = renderHook(() => useHealthData());

    act(() => {
      result.current.fetchWeightHistory(30);
    });

    expect(result.current.isLoading).toBe(true);

    await act(async () => {
      resolvePromise({ data: [] });
    });

    expect(result.current.isLoading).toBe(false);
  });

  // ----- T2.6 Błąd sieciowy -----
  it('T2.6: should handle API errors gracefully', async () => {
    mockApi.get.mockRejectedValue(new Error('Server Error'));

    const { result } = renderHook(() => useHealthData());

    await act(async () => {
      await result.current.fetchWeightHistory(30);
    });

    expect(result.current.error).toBe('Server Error');
    expect(result.current.weightHistory).toEqual([]);
    expect(result.current.isLoading).toBe(false);
  });
});
