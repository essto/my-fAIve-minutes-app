import React from 'react';
import { render } from '@testing-library/react-native';
import { SparklineChart } from '../SparklineChart';

describe('SparklineChart', () => {
  const defaultData = [
    { value: 65, date: '2023-01-01' },
    { value: 68, date: '2023-01-02' },
    { value: 64, date: '2023-01-03' },
  ];

  it('SC1: should render the SVG chart', () => {
    const { getByTestId } = render(<SparklineChart data={defaultData} color="#10B981" />);
    expect(getByTestId('sparkline-svg')).toBeTruthy();
  });

  it('SC2: should handle empty data without crashing', () => {
    const { queryByTestId } = render(<SparklineChart data={[]} />);
    expect(queryByTestId('sparkline-path')).toBeNull(); // Shouldn't render path
  });
});
