import 'reflect-metadata';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import request from 'supertest';
import { describe, it, expect, beforeAll, vi } from 'vitest';
import { WeightModule } from '../weight.module';
import { ZodValidationPipe } from '../../../../packages/nest-utils/src/index';
// Using WeightController via WeightModule, but we mock the internal WeightService
// to isolate the Outer Loop API testing boundary.

describe('POST /weight (body composition) - Outer Loop Contract', () => {
  let app: INestApplication;
  const mockWeightService = {
    addReading: vi.fn(),
    getHistory: vi.fn(),
    calculateBMI: vi.fn(),
    getHealthSummary: vi.fn(),
  };

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [WeightModule],
    })
      .overrideProvider('WEIGHT_SERVICE')
      .useValue(mockWeightService)
      .compile();

    app = moduleFixture.createNestApplication();
    await app.init();
  });

  it('accepts body-composition payload and returns 201 with bmi', async () => {
    const payload = {
      value: 82.3,
      unit: 'kg',
      bmi: 26.1,
      fatPercent: 22.5,
      muscleMassKg: 38.1,
      waterPercent: 56.2,
      metabolicAge: 40,
    };

    mockWeightService.addReading.mockResolvedValue({
      id: 'mock-id-xyz',
      ...payload,
      userId: 'u1',
      timestamp: new Date(),
    });

    const res = await request(app.getHttpServer()).post('/weight').send(payload);

    expect(res.status).toBe(201);
    expect(res.body.bmi).toBe(26.1);
    expect(res.body.fatPercent).toBe(22.5);

    // contract: service should receive the expanded payload, not stripped by Zod
    expect(mockWeightService.addReading).toHaveBeenCalledWith(
      'u1',
      expect.objectContaining({
        bmi: 26.1,
        fatPercent: 22.5,
      }),
    );
  });

  describe('GET /weight', () => {
    it('returns readings with bmi and fatPercent', async () => {
      mockWeightService.getHistory.mockResolvedValue([
        {
          id: '1',
          value: 82.3,
          unit: 'kg',
          bmi: 26.1,
          fatPercent: 22.5,
          userId: 'u1',
          timestamp: new Date(),
        },
      ]);
      const res = await request(app.getHttpServer()).get('/weight');

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body)).toBe(true);
      expect(res.body[0]).toHaveProperty('bmi');
      expect(res.body[0]).toHaveProperty('fatPercent');
    });
  });
});
