import { renderHook, act, waitFor } from '@testing-library/react-native';
import { useDietData } from '../useDietData';
import api from '../../services/api';

jest.mock('../../services/api');
const mockApi = api as jest.Mocked<typeof api>;

describe('useDietData', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ----- T3.1 Pobranie posiłków dnia -----
  it('T3.1: should fetch meals for given date', async () => {
    mockApi.get.mockResolvedValue({
      data: [{ id: 'm1', name: 'Śniadanie', consumedAt: '2026-02-23T08:00:00Z', products: [] }],
    });

    const { result } = renderHook(() => useDietData());

    await act(async () => {
      await result.current.fetchMeals('2026-02-23');
    });

    await waitFor(() => {
      expect(result.current.meals).toHaveLength(1);
      expect(result.current.meals[0].name).toBe('Śniadanie');
    });
  });

  // ----- T3.2 Dodanie posiłku -----
  it('T3.2: should log a new meal', async () => {
    mockApi.post.mockResolvedValue({
      data: { id: 'm2', name: 'Obiad', consumedAt: '2026-02-23T13:00:00Z' },
    });

    const { result } = renderHook(() => useDietData());

    await act(async () => {
      await result.current.logMeal({
        name: 'Obiad',
        products: [{ name: 'Ryż', calories: 200, protein: 5, carbs: 45, fat: 1, quantity: 150 }],
      });
    });

    expect(mockApi.post).toHaveBeenCalledWith(
      '/diet/meals',
      expect.objectContaining({ name: 'Obiad' }),
    );
  });

  // ----- T3.3 Barcode lookup — sukces -----
  it('T3.3: should find product by barcode', async () => {
    mockApi.get.mockResolvedValue({
      data: { name: 'Jogurt naturalny', calories: 80, protein: 5, carbs: 8, fat: 3 },
    });

    const { result } = renderHook(() => useDietData());

    let product;
    await act(async () => {
      product = await result.current.lookupBarcode('5901234123457');
    });

    expect(product).toHaveProperty('name', 'Jogurt naturalny');
  });

  // ----- T3.4 Barcode lookup — nie znaleziono -----
  it('T3.4: should return null when barcode not found', async () => {
    mockApi.get.mockRejectedValue({ response: { status: 404 } });

    const { result } = renderHook(() => useDietData());

    await act(async () => {
      await result.current.lookupBarcode('0000000000000');
    });

    expect(result.current.barcodeResult).toBeNull();
    expect(result.current.barcodeError).toBe('Produkt nie znaleziony');
  });

  // ----- T3.5 Dzienne podsumowanie kalorii -----
  it('T3.5: should calculate daily calorie summary', async () => {
    mockApi.get.mockResolvedValue({
      data: {
        total: { calories: 1800, protein: 80, carbs: 200, fat: 60 },
        isDeficit: false,
        isSurplus: false,
      },
    });

    const { result } = renderHook(() => useDietData());

    await act(async () => {
      await result.current.fetchDailySummary('2026-02-23');
    });

    await waitFor(() => {
      expect(result.current.dailySummary?.total.calories).toBe(1800);
    });
  });

  // ----- T3.6 Walidacja posiłku przed wysłaniem -----
  it('T3.6: should validate meal data with Zod schema before POST', async () => {
    const { result } = renderHook(() => useDietData());

    // GIVEN: puste products (should fail validation)
    await act(async () => {
      await result.current.logMeal({ name: '', products: [] });
    });

    // THEN: nie powinno wysłać request, powinien pojawić się błąd walidacji
    expect(mockApi.post).not.toHaveBeenCalled();
    expect(result.current.validationError).toBeTruthy();
  });
});
