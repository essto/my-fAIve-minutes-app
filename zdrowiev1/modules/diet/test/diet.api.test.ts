import 'reflect-metadata';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import request from 'supertest';
import { describe, it, expect, beforeAll, vi } from 'vitest';
import { DietModule } from '../diet.module';
import { DietService } from '../domain/services/diet.service';

describe('DietController (API)', () => {
  let app: INestApplication;
  const mockService = {
    addEntry: vi.fn(),
    getHistory: vi.fn(),
  };

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [DietModule],
    })
      .overrideProvider('DIET_SERVICE')
      .useValue(mockService)
      .compile();

    app = moduleFixture.createNestApplication();
    await app.init();
  });

  it('POST /diet - Success', async () => {
    const payload = {
      name: 'Chicken Salad',
      calories: 350,
      protein: 25,
      carbs: 10,
      fat: 15,
    };
    mockService.addEntry.mockResolvedValue({ id: '1', ...payload, userId: 'u1' });

    const response = await request(app.getHttpServer()).post('/diet').send(payload);

    expect(response.status).toBe(201);
    expect(response.body.name).toBe('Chicken Salad');
  });

  it('POST /diet - Validation Error', async () => {
    const payload = { calories: -10 }; // Invalid calories

    const response = await request(app.getHttpServer()).post('/diet').send(payload);

    expect(response.status).toBe(400);
  });

  it('GET /diet - Success', async () => {
    mockService.getHistory.mockResolvedValue([
      { id: '1', name: 'Chicken Salad', calories: 350, userId: 'u1' },
    ]);

    const response = await request(app.getHttpServer()).get('/diet');

    expect(response.status).toBe(200);
    expect(Array.isArray(response.body)).toBe(true);
  });
});
