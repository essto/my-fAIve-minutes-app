import 'reflect-metadata';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import request from 'supertest';
import { describe, it, expect, beforeAll, vi } from 'vitest';
import { DiagnosisModule } from '../diagnosis.module';
import { DiagnosisService } from '../domain/services/diagnosis.service';

describe('DiagnosisController (API)', () => {
  let app: INestApplication;
  const mockService = {
    reportSymptoms: vi.fn(),
    getHistory: vi.fn(),
  };

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [DiagnosisModule],
    })
      .overrideProvider('DIAGNOSIS_SERVICE')
      .useValue(mockService)
      .compile();

    app = moduleFixture.createNestApplication();
    await app.init();
  });

  it('POST /diagnosis/report - Success', async () => {
    const payload = { description: 'Headache and fatigue', severity: 4 };
    mockService.reportSymptoms.mockResolvedValue({ id: '1', ...payload, userId: 'u1' });

    const response = await request(app.getHttpServer()).post('/diagnosis/report').send(payload);

    expect(response.status).toBe(201);
    expect(response.body.description).toBe('Headache and fatigue');
  });

  it('POST /diagnosis/report - Validation Error', async () => {
    const payload = { severity: 11 }; // Invalid severity > 10

    const response = await request(app.getHttpServer()).post('/diagnosis/report').send(payload);

    expect(response.status).toBe(400);
  });

  it('GET /diagnosis/history - Success', async () => {
    mockService.getHistory.mockResolvedValue([
      { id: '1', description: 'Headache', severity: 4, userId: 'u1' },
    ]);

    const response = await request(app.getHttpServer()).get('/diagnosis/history');

    expect(response.status).toBe(200);
    expect(Array.isArray(response.body)).toBe(true);
  });
});
