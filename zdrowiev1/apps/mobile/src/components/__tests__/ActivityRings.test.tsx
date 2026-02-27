import React from 'react';
import { render } from '@testing-library/react-native';
import { ActivityRings } from '../ActivityRings';

describe('ActivityRings', () => {
  const defaultProps = {
    stepsProgress: 0.8,
    caloriesProgress: 0.6,
    exerciseProgress: 0.4,
  };

  it('AR1: should render all three rings', () => {
    const { getByTestId } = render(<ActivityRings {...defaultProps} />);
    expect(getByTestId('ring-steps')).toBeTruthy();
    expect(getByTestId('ring-calories')).toBeTruthy();
    expect(getByTestId('ring-exercise')).toBeTruthy();
  });

  it('AR2: should show icon/labels inside', () => {
    const { getByText } = render(<ActivityRings {...defaultProps} />);
    // According to plan, we test for icons or something recognizable
    // E.g., a central icon, here just checking if it renders text if it has one or just testing the presence of SVG
    expect(getByText('🏃')).toBeTruthy();
  });

  it('AR3: should handle empty progress gracefully (0)', () => {
    const { getByTestId } = render(
      <ActivityRings stepsProgress={0} caloriesProgress={0} exerciseProgress={0} />,
    );
    expect(getByTestId('ring-steps')).toBeTruthy();
    expect(getByTestId('ring-calories')).toBeTruthy();
    expect(getByTestId('ring-exercise')).toBeTruthy();
  });
});
