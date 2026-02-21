import 'reflect-metadata';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import request from 'supertest';
import { describe, it, expect, beforeAll, vi } from 'vitest';
import { WeightModule } from '../weight.module';
import { WeightService } from '../domain/services/weight.service';

describe('WeightController (API)', () => {
  let app: INestApplication;
  const mockWeightService = {
    addReading: vi.fn(),
    getHistory: vi.fn(),
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

  it('POST /weight - Success', async () => {
    const payload = { value: 75.5, unit: 'kg', source: 'manual' };
    mockWeightService.addReading.mockResolvedValue({ id: '1', ...payload, userId: 'u1' });

    const response = await request(app.getHttpServer()).post('/weight').send(payload);

    expect(response.status).toBe(201);
    expect(response.body).toHaveProperty('id');
    expect(response.body.value).toBe(75.5);
  });

  it('POST /weight - Validation Error (too low)', async () => {
    const payload = { value: 0.1, unit: 'kg' };

    const response = await request(app.getHttpServer()).post('/weight').send(payload);

    expect(response.status).toBe(400);
  });

  it('GET /weight - Success', async () => {
    mockWeightService.getHistory.mockResolvedValue([
      { id: '1', value: 75.5, unit: 'kg', userId: 'u1' },
    ]);

    const response = await request(app.getHttpServer()).get('/weight');

    expect(response.status).toBe(200);
    expect(Array.isArray(response.body)).toBe(true);
    expect(response.body[0].value).toBe(75.5);
  });
});
