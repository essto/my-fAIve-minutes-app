import React from 'react';
import { render } from '@testing-library/react-native';
import { RootNavigator, TabNavigator, AuthStack } from '../RootNavigator';
import { useAuth } from '../../hooks/useAuth';

jest.mock('../../hooks/useAuth');
const mockUseAuth = useAuth as jest.Mock;

jest.mock('@react-navigation/native', () => {
  return {
    NavigationContainer: ({ children }: any) => <>{children}</>,
    DarkTheme: { colors: {} },
    DefaultTheme: { colors: {} },
  };
});

jest.mock('@react-navigation/native-stack', () => {
  const React = require('react');
  const { View } = require('react-native');
  return {
    createNativeStackNavigator: () => ({
      Navigator: ({ children }: any) => <>{children}</>,
      Screen: ({ name }: any) => <View testID={`screen-${name}`} />,
    }),
  };
});

jest.mock('@react-navigation/bottom-tabs', () => {
  const React = require('react');
  const { View } = require('react-native');
  return {
    createBottomTabNavigator: () => ({
      Navigator: ({ children }: any) => <>{children}</>,
      Screen: ({ name }: any) => <View testID={`tab-${name}`} />,
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
