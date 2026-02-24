import React from 'react';
import { render } from '@testing-library/react-native';
import { RootNavigator, TabNavigator, AuthStack } from '../RootNavigator';
import { useAuth } from '../../hooks/useAuth';

jest.mock('../../hooks/useAuth');
const mockUseAuth = useAuth as jest.Mock;

jest.mock('@react-navigation/native', () => {
  return {
    NavigationContainer: ({ children }: any) => <>{children}</>,
  };
});

jest.mock('@react-navigation/native-stack', () => {
  return {
    createNativeStackNavigator: () => ({
      Navigator: ({ children }: any) => <mock-stack-navigator>{children}</mock-stack-navigator>,
      Screen: ({ name }: any) => <mock-stack-screen testID={`screen-${name}`} />,
    }),
  };
});

jest.mock('@react-navigation/bottom-tabs', () => {
  return {
    createBottomTabNavigator: () => ({
      Navigator: ({ children }: any) => <mock-tab-navigator>{children}</mock-tab-navigator>,
      Screen: ({ name }: any) => <mock-tab-screen testID={`tab-${name}`} />,
    }),
  };
});

describe('RootNavigator', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('R1.1: should render loading indicator when isLoading is true', () => {
    mockUseAuth.mockReturnValue({
      isLoading: true,
      isAuthenticated: false,
    });

    const { getByTestId } = render(<RootNavigator />);
    expect(getByTestId('root-loading')).toBeTruthy();
  });

  it('R1.2: should render AuthStack when isAuthenticated is false', () => {
    mockUseAuth.mockReturnValue({
      isLoading: false,
      isAuthenticated: false,
    });

    const { getByTestId } = render(<RootNavigator />);
    expect(getByTestId('screen-Login')).toBeTruthy();
  });

  it('R1.3: should render TabNavigator when isAuthenticated is true', () => {
    mockUseAuth.mockReturnValue({
      isLoading: false,
      isAuthenticated: true,
    });

    const { getByTestId } = render(<RootNavigator />);
    expect(getByTestId('tab-Home')).toBeTruthy();
    expect(getByTestId('tab-Health')).toBeTruthy();
    expect(getByTestId('tab-Diet')).toBeTruthy();
  });
});
