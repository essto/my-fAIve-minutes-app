import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { DietScreen } from '../DietScreen';
import { useDietData } from '../../hooks/useDietData';

jest.mock('../../hooks/useDietData');
const mockUseDietData = useDietData as jest.Mock;

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
  });

  it('D1.1: should render daily summary', () => {
    const { getByText } = render(<DietScreen />);
    expect(getByText('Podsumowanie Dnia')).toBeTruthy();
    expect(getByText('Kalorie: 0 kcal')).toBeTruthy();
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
});
