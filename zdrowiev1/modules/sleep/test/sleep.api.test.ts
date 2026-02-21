import 'reflect-metadata';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import request from 'supertest';
import { describe, it, expect, beforeAll, vi } from 'vitest';
import { SleepModule } from '../sleep.module';
import { SleepService } from '../domain/services/sleep.service';

describe('SleepController (API)', () => {
  let app: INestApplication;
  const mockService = {
    addRecord: vi.fn(),
    getHistory: vi.fn(),
  };

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [SleepModule],
    })
      .overrideProvider('SLEEP_SERVICE')
      .useValue(mockService)
      .compile();

    app = moduleFixture.createNestApplication();
    await app.init();
  });

  it('POST /sleep - Success', async () => {
    const payload = {
      startTime: new Date('2024-01-01T22:00:00Z').toISOString(),
      endTime: new Date('2024-01-02T06:00:00Z').toISOString(),
      quality: 8,
    };
    mockService.addRecord.mockResolvedValue({ id: '1', ...payload, userId: 'u1' });

    const response = await request(app.getHttpServer()).post('/sleep').send(payload);

    expect(response.status).toBe(201);
    expect(response.body.quality).toBe(8);
  });

  it('POST /sleep - Validation Error', async () => {
    const payload = { quality: 11 }; // Invalid quality > 10

    const response = await request(app.getHttpServer()).post('/sleep').send(payload);

    expect(response.status).toBe(400);
  });

  it('GET /sleep - Success', async () => {
    mockService.getHistory.mockResolvedValue([{ id: '1', quality: 8, userId: 'u1' }]);

    const response = await request(app.getHttpServer()).get('/sleep');

    expect(response.status).toBe(200);
    expect(Array.isArray(response.body)).toBe(true);
  });
});
