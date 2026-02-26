import React from 'react';
import { render } from '@testing-library/react-native';
import { MetricCard } from '../MetricCard';

describe('MetricCard Component', () => {
  it('renders label, value and unit correctly', () => {
    const { getByText } = render(<MetricCard label="Current Weight" value="75.5" unit="kg" />);

    expect(getByText('Current Weight')).toBeTruthy();
    expect(getByText('75.5')).toBeTruthy();
    expect(getByText('kg')).toBeTruthy();
  });
});
