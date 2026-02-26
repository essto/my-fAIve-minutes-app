import React from 'react';
import { render } from '@testing-library/react-native';
import { LoadingScreen } from '../LoadingScreen';

describe('LoadingScreen Component', () => {
  it('renders ActivityIndicator with given testID', () => {
    const { getByTestId } = render(<LoadingScreen testID="custom-loading" />);
    expect(getByTestId('custom-loading')).toBeTruthy();
  });
});
