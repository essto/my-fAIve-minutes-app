import React from 'react';
import { render } from '@testing-library/react-native';
import App from '../../App';

jest.mock('../hooks/useAuth', () => ({
  useAuth: jest.fn().mockReturnValue({
    isAuthenticated: false,
    isLoading: true,
  }),
}));

describe('App', () => {
  it('renders correctly', () => {
    const component = render(<App />);
    expect(component).toBeTruthy();
  });
});
