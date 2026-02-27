import React from 'react';
import { render } from '@testing-library/react-native';
import { NotificationBell } from '../NotificationBell';

describe('NotificationBell', () => {
  it('NB1: should render the bell icon', () => {
    const { getByText } = render(<NotificationBell count={0} />);
    expect(getByText('🔔')).toBeTruthy();
  });

  it('NB2: should show the red dot if count > 0', () => {
    const { getByTestId, getByText } = render(<NotificationBell count={3} />);
    expect(getByTestId('notification-dot')).toBeTruthy();
    expect(getByText('3')).toBeTruthy();
  });

  it('NB3: should hide the red dot if count === 0', () => {
    const { queryByTestId } = render(<NotificationBell count={0} />);
    expect(queryByTestId('notification-dot')).toBeNull();
  });
});
