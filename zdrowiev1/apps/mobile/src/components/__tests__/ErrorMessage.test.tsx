import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { ErrorMessage } from '../ErrorMessage';

describe('ErrorMessage Component', () => {
  it('renders error text correctly', () => {
    const { getByText } = render(<ErrorMessage message="Something went wrong" />);
    expect(getByText('Something went wrong')).toBeTruthy();
  });

  it('renders retry button and handles click', () => {
    const onRetryMock = jest.fn();
    const { getByText } = render(<ErrorMessage message="Error" onRetry={onRetryMock} />);

    const button = getByText('Ponów');
    expect(button).toBeTruthy();

    fireEvent.press(button);
    expect(onRetryMock).toHaveBeenCalledTimes(1);
  });
});
