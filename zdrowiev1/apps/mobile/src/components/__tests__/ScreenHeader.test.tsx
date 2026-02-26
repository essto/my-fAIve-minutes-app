import React from 'react';
import { render } from '@testing-library/react-native';
import { ScreenHeader } from '../ScreenHeader';

describe('ScreenHeader Component', () => {
  it('renders title correctly', () => {
    const { getByText } = render(<ScreenHeader title="Dashboard" />);
    expect(getByText('Dashboard')).toBeTruthy();
  });
});
