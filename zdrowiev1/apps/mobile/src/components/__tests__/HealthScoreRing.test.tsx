import React from 'react';
import { render } from '@testing-library/react-native';
import { processColor } from 'react-native';
import { HealthScoreRing } from '../HealthScoreRing';

describe('HealthScoreRing', () => {
  it('HSR1: should render score number inside the ring', () => {
    const { getByText } = render(<HealthScoreRing score={85} />);
    expect(getByText('85')).toBeTruthy();
  });

  it('HSR2: should show green color for score >= 80', () => {
    const { getByTestId } = render(<HealthScoreRing score={85} />);
    const ring = getByTestId('health-ring-arc');
    const color =
      ring.props.stroke.payload !== undefined ? ring.props.stroke.payload : ring.props.stroke;
    expect(color).toEqual(processColor('#10B981')); // green
  });

  it('HSR3: should show yellow color for score 50-79', () => {
    const { getByTestId } = render(<HealthScoreRing score={65} />);
    const ring = getByTestId('health-ring-arc');
    const color =
      ring.props.stroke.payload !== undefined ? ring.props.stroke.payload : ring.props.stroke;
    expect(color).toEqual(processColor('#F59E0B')); // amber/yellow
  });

  it('HSR4: should show red color for score < 50', () => {
    const { getByTestId } = render(<HealthScoreRing score={30} />);
    const ring = getByTestId('health-ring-arc');
    const color =
      ring.props.stroke.payload !== undefined ? ring.props.stroke.payload : ring.props.stroke;
    expect(color).toEqual(processColor('#EF4444')); // red
  });
});
