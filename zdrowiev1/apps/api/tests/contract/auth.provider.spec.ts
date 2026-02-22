import { describe, it, beforeAll, afterAll } from 'vitest';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import { Verifier } from '@pact-foundation/pact';
import { AppModule } from '../../src/app.module';
import * as path from 'path';

describe('Pact Provider Verification', () => {
  let app: INestApplication;
  let serverUrl: string;

  beforeAll(async () => {
    process.env.JWT_SECRET = 'pact-test-secret';
    process.env.POSTGRES_HOST = 'localhost';
    process.env.POSTGRES_PORT = '5432';
    process.env.POSTGRES_USER = 'postgres';
    process.env.POSTGRES_PASSWORD = 'postgres';
    process.env.POSTGRES_DB = 'health';

    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    app.setGlobalPrefix('api');

    // Log errors to console
    app.useGlobalFilters({
      catch(exception: any) {
        console.error('SERVER ERROR DURING PACT:', exception);
        throw exception;
      },
    } as any);

    await app.listen(0);
    serverUrl = await app.getUrl();
    console.log(`Pact Provider Mock Server running at ${serverUrl}`);
  }, 30000);

  afterAll(async () => {
    await app.close();
  });

  it('powinien przejść weryfikację kontraktu web-api', async () => {
    const verifier = new Verifier({
      provider: 'api',
      providerBaseUrl: serverUrl,
      pactUrls: [path.resolve(process.cwd(), '../../pacts/web-api.json')],
      stateHandlers: {
        'użytkownik demo istnieje': async () => {
          // State handler if needed (user already seeded by SeedDemoService)
          return Promise.resolve('User demo exists');
        },
      },
    });

    await verifier.verifyProvider();
  }, 60000);
});
