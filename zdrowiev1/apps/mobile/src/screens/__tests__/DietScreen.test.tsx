import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { DietScreen } from '../DietScreen';
import { useDietData } from '../../hooks/useDietData';
import { useBarcode } from '../../hooks/useBarcode';

jest.mock('../../hooks/useDietData');
jest.mock('../../hooks/useBarcode');
const mockUseDietData = useDietData as jest.Mock;
const mockUseBarcode = useBarcode as jest.Mock;

describe('DietScreen', () => {
  const mockFetchMeals = jest.fn();
  const mockLogMeal = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseDietData.mockReturnValue({
      meals: [],
      dailySummary: {
        total: { calories: 0, protein: 0, carbs: 0, fat: 0 },
      },
      isLoading: false,
      fetchMeals: mockFetchMeals,
      logMeal: mockLogMeal,
      fetchDailySummary: jest.fn(),
    });
    mockUseBarcode.mockReturnValue({
      isScanning: false,
      product: null,
      isLoading: false,
      error: null,
      scanBarcode: jest.fn(),
      cancelScan: jest.fn(),
      lookupProduct: jest.fn(),
    });
  });

  it('D1.1: should render daily summary', () => {
    const { getByText } = render(<DietScreen />);
    expect(getByText('Podsumowanie Dnia')).toBeTruthy();
    expect(getByText('0')).toBeTruthy();
    expect(getByText('kcal')).toBeTruthy();
  });

  it('D1.2: should display meals list when available', () => {
    mockUseDietData.mockReturnValue({
      ...mockUseDietData(),
      meals: [{ id: '1', name: 'Śniadanie', consumedAt: '2026-02-23', products: [] }],
    });

    const { getByText } = render(<DietScreen />);
    expect(getByText('Śniadanie')).toBeTruthy();
  });

  it('D1.3: should update new meal name input', () => {
    const { getByPlaceholderText } = render(<DietScreen />);
    const input = getByPlaceholderText('Nazwa posiłku (np. Obiad)');

    fireEvent.changeText(input, 'Kolacja');
    expect(input.props.value).toBe('Kolacja');
  });

  it('D1.4: should call logMeal when add button is pressed', async () => {
    const { getByPlaceholderText, getByText } = render(<DietScreen />);

    fireEvent.changeText(getByPlaceholderText('Nazwa posiłku (np. Obiad)'), 'Kolacja');
    fireEvent.press(getByText('Dodaj posiłek'));

    await waitFor(() => {
      expect(mockLogMeal).toHaveBeenCalledWith(expect.objectContaining({ name: 'Kolacja' }));
    });
  });

  it('D1.5: should show loading indicator when fetching', () => {
    mockUseDietData.mockReturnValue({
      ...mockUseDietData(),
      isLoading: true,
    });

    const { getByTestId } = render(<DietScreen />);
    expect(getByTestId('diet-loading')).toBeTruthy();
  });

  it('D1.6: should open barcode scanner when scan button is pressed', () => {
    const mockScanBarcode = jest.fn();
    mockUseBarcode.mockReturnValue({
      isScanning: false,
      product: null,
      isLoading: false,
      error: null,
      scanBarcode: mockScanBarcode,
      cancelScan: jest.fn(),
      lookupProduct: jest.fn(),
    });

    const { getByTestId } = render(<DietScreen />);
    fireEvent.press(getByTestId('scan-barcode-button'));
    expect(mockScanBarcode).toHaveBeenCalled();
  });

  it('D1.7: should display BarcodeScanner component when isScanning is true', () => {
    mockUseBarcode.mockReturnValue({
      isScanning: true,
      product: null,
      isLoading: false,
      error: null,
      scanBarcode: jest.fn(),
      cancelScan: jest.fn(),
      lookupProduct: jest.fn(),
    });

    const { getByTestId } = render(<DietScreen />);
    expect(getByTestId('camera-view')).toBeTruthy(); // from mocked BarcodeScanner
  });
});
