import React from 'react';
import { View } from 'react-native';
import Svg, { Path } from 'react-native-svg';

export interface SparklineDataPoint {
  value: number;
  date: string;
}

interface SparklineChartProps {
  data: SparklineDataPoint[];
  color?: string;
  width?: number;
  height?: number;
  strokeWidth?: number;
}

export const SparklineChart = ({
  data,
  color = '#8251EE',
  width = 100,
  height = 40,
  strokeWidth = 2,
}: SparklineChartProps) => {
  if (!data || data.length === 0) {
    return <View style={{ width, height }} />;
  }

  if (data.length === 1) {
    return (
      <View
        testID="sparkline-svg"
        className="justify-center items-center"
        style={{ width, height }}
      >
        <View
          style={{
            width: strokeWidth * 2,
            height: strokeWidth * 2,
            borderRadius: strokeWidth,
            backgroundColor: color,
          }}
        />
      </View>
    );
  }

  const values = data.map((d) => d.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1; // Prevent division by zero

  // Create path
  const padding = strokeWidth;
  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;

  const points = values.map((val, index) => {
    const x = padding + (index / (values.length - 1)) * innerWidth;
    const y = padding + innerHeight - ((val - min) / range) * innerHeight;
    return `${x},${y}`;
  });

  const pathD = `M ${points.join(' L ')}`;

  return (
    <View style={{ width, height }}>
      <Svg testID="sparkline-svg" width={width} height={height}>
        <Path
          testID="sparkline-path"
          d={pathD}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </Svg>
    </View>
  );
};
