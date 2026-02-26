import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { OcrScreen } from '../OcrScreen';
import { useOCR } from '../../hooks/useOCR';

jest.mock('../../hooks/useOCR');
const mockUseOCR = useOCR as jest.Mock;

describe('OcrScreen', () => {
  const mockTakePhoto = jest.fn();
  const mockPickFromGallery = jest.fn();
  const mockClearResult = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseOCR.mockReturnValue({
      ocrResult: null,
      isLoading: false,
      error: null,
      takePhoto: mockTakePhoto,
      pickFromGallery: mockPickFromGallery,
      clearResult: mockClearResult,
    });
  });

  it('OS1.1: should render action buttons initially', () => {
    const { getByText } = render(<OcrScreen />);
    expect(getByText('Uruchom Aparat')).toBeTruthy();
    expect(getByText('Wybierz z Galerii')).toBeTruthy();
  });

  it('OS1.2: should call takePhoto when Camera button is pressed', () => {
    const { getByText } = render(<OcrScreen />);
    fireEvent.press(getByText('Uruchom Aparat'));
    expect(mockTakePhoto).toHaveBeenCalled();
  });

  it('OS1.3: should call pickFromGallery when Gallery button is pressed', () => {
    const { getByText } = render(<OcrScreen />);
    fireEvent.press(getByText('Wybierz z Galerii'));
    expect(mockPickFromGallery).toHaveBeenCalled();
  });

  it('OS1.4: should show loading indicator when processing', () => {
    mockUseOCR.mockReturnValue({
      ...mockUseOCR(),
      isLoading: true,
    });
    const { getByTestId } = render(<OcrScreen />);
    expect(getByTestId('ocr-loading')).toBeTruthy();
  });

  it('OS1.5: should display error message if failed', () => {
    mockUseOCR.mockReturnValue({
      ...mockUseOCR(),
      error: 'Uprawnienia odmówione.',
    });
    const { getByTestId } = render(<OcrScreen />);
    expect(getByTestId('ocr-error')).toBeTruthy();
  });

  it('OS1.6: should display OCR result and allow clearing', () => {
    mockUseOCR.mockReturnValue({
      ...mockUseOCR(),
      ocrResult: { text: 'Paracetamol', confidence: 0.98 },
    });
    const { getByText, queryByText } = render(<OcrScreen />);

    // Check results are rendered
    expect(getByText(/Rozpoznany Tekst/i)).toBeTruthy();
    expect(getByText('Paracetamol')).toBeTruthy();

    // Check action buttons are hidden
    expect(queryByText('Uruchom Aparat')).toBeNull();

    // Clear
    fireEvent.press(getByText('Skanuj ponownie'));
    expect(mockClearResult).toHaveBeenCalled();
  });
});
