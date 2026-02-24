import 'reflect-metadata';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import request from 'supertest';
import { describe, it, expect, beforeAll, vi } from 'vitest';
import { VisualizationModule } from '../visualization.module';

describe('GET /visualization/health-details (Outer Loop Contract)', () => {
  let app: INestApplication;

  const mockWeightService = {
    getHistory: vi.fn(),
    getHealthSummary: vi.fn(),
  };

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [VisualizationModule],
    })
      .overrideProvider('WEIGHT_SERVICE')
      .useValue(mockWeightService)
      .compile();

    app = moduleFixture.createNestApplication();
    await app.init();
  });

  it('returns metrics.weight and charts.weightHistory populated from real service', async () => {
    // Return mock data mimicking a real DB response
    mockWeightService.getHistory.mockResolvedValue([
      { value: 83.1, bmi: 26.5, timestamp: new Date('2024-03-15T12:00:00Z') }, // older
      { value: 82.5, bmi: 26.3, timestamp: new Date('2024-06-15T12:00:00Z') }, // latest
    ]);

    mockWeightService.getHealthSummary.mockResolvedValue({
      current: 82.5,
      change30d: -0.6,
      bmi: 26.3,
    });

    const res = await request(app.getHttpServer()).get('/visualization/health-details');

    expect(res.status).toBe(200);

    // Assert metrics
    expect(res.body.metrics.weight.current).toBe(82.5);
    expect(res.body.metrics.weight.change30d).toBe(-0.6);
    expect(res.body.metrics.weight.bmi).toBe(26.3);

    // Assert charts format
    expect(Array.isArray(res.body.charts.weightHistory.data)).toBe(true);
    expect(res.body.charts.weightHistory.colors).toContain('#10b981');
    expect(res.body.charts.weightHistory.data[1].value).toBe(82.5);
    expect(res.body.charts.weightHistory.data[1].label).toBe('2024-06-15');
  });
});
