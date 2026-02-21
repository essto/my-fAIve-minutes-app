import 'reflect-metadata';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import request from 'supertest';
import { describe, it, expect, beforeAll, vi } from 'vitest';
import { HeartRateModule } from '../heart-rate.module';
import { HeartRateService } from '../domain/services/heart-rate.service';

describe('HeartRateController (API)', () => {
  let app: INestApplication;
  const mockService = {
    addReading: vi.fn(),
    getHistory: vi.fn(),
  };

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [HeartRateModule],
    })
      .overrideProvider('HEART_RATE_SERVICE')
      .useValue(mockService)
      .compile();

    app = moduleFixture.createNestApplication();
    await app.init();
  });

  it('POST /heart-rate - Success', async () => {
    const payload = { value: 72, unit: 'bpm' };
    mockService.addReading.mockResolvedValue({ id: '1', ...payload, userId: 'u1' });

    const response = await request(app.getHttpServer()).post('/heart-rate').send(payload);

    expect(response.status).toBe(201);
    expect(response.body.value).toBe(72);
  });

  it('POST /heart-rate - Validation Error', async () => {
    const payload = { value: -1 }; // Invalid HR

    const response = await request(app.getHttpServer()).post('/heart-rate').send(payload);

    expect(response.status).toBe(400);
  });

  it('GET /heart-rate - Success', async () => {
    mockService.getHistory.mockResolvedValue([{ id: '1', value: 72, unit: 'bpm', userId: 'u1' }]);

    const response = await request(app.getHttpServer()).get('/heart-rate');

    expect(response.status).toBe(200);
    expect(Array.isArray(response.body)).toBe(true);
  });
});
