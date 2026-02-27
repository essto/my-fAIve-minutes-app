import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { BarcodeScanner } from '../BarcodeScanner';
import { View } from 'react-native';

// Mock expo-camera
jest.mock('expo-camera', () => {
  const React = require('react');
  const View = require('react-native').View;

  return {
    useCameraPermissions: () => [{ granted: true, status: 'granted' }, jest.fn()],
    CameraView: ({ onBarcodeScanned, testID, children }: any) => {
      // Create a mock ref allowing calling onBarcodeScanned manually for testing
      return (
        <View testID={testID} onLayout={() => onBarcodeScanned({ data: '5901234' })}>
          {children}
        </View>
      );
    },
  };
});

describe('BarcodeScanner', () => {
  const mockOnScan = jest.fn();
  const mockOnClose = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('B2.1: should render camera viewfinder', () => {
    const { getByTestId } = render(
      <BarcodeScanner onBarcodeScanned={mockOnScan} onClose={mockOnClose} />,
    );
    expect(getByTestId('barcode-viewfinder')).toBeTruthy();
    expect(getByTestId('camera-view')).toBeTruthy();
  });

  it('B2.2: should call onBarcodeScanned callback', () => {
    const { getByTestId } = render(
      <BarcodeScanner onBarcodeScanned={mockOnScan} onClose={mockOnClose} />,
    );
    // Our mock triggers onBarcodeScanned on layout
    fireEvent(getByTestId('camera-view'), 'layout');
    expect(mockOnScan).toHaveBeenCalledWith('5901234');
  });

  it('B2.3: should call onClose when cancel button is pressed', () => {
    const { getByText } = render(
      <BarcodeScanner onBarcodeScanned={mockOnScan} onClose={mockOnClose} />,
    );
    fireEvent.press(getByText('Anuluj'));
    expect(mockOnClose).toHaveBeenCalled();
  });
});
