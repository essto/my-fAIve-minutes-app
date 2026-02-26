import React from 'react';
import { render } from '@testing-library/react-native';
import { GlassCard } from '../GlassCard';
import { Text } from 'react-native';

describe('GlassCard Component', () => {
  it('renders children correctly', () => {
    const { getByText } = render(
      <GlassCard>
        <Text>Hello GlassCard</Text>
      </GlassCard>,
    );

    expect(getByText('Hello GlassCard')).toBeTruthy();
  });

  it('applies custom className', () => {
    const { getByTestId } = render(
      <GlassCard testID="glass-card" className="mt-4">
        <Text>Content</Text>
      </GlassCard>,
    );

    const card = getByTestId('glass-card');
    // Minimalny check propsów stylizujących.
    // Ponieważ NativeWind transformuje className w style, trudno testować wyjściowy styl w prosty sposób w JEST bez pełnego setupu.
    // Ale możemy sprawdzić className jeśli przekazujemy go dalej (w zależności od implementacji).
    expect(card.props.className).toContain('mt-4');
  });
});
