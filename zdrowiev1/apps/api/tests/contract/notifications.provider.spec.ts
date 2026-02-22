import { describe, it, beforeAll, afterAll } from 'vitest';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import { Verifier } from '@pact-foundation/pact';
import { AppModule } from '../../src/app.module';
import { JwtAuthGuard } from '@monorepo/auth';
import { NotificationService } from '../../../../modules/notifications/domain/notification.service';
import {
  Notification,
  NotificationType,
  NotificationChannel,
} from '../../../../modules/notifications/domain/notification.entity';
import * as path from 'path';

describe('Notifications Pact Provider Verification', () => {
  let app: INestApplication;
  let serverUrl: string;
  let notificationService: NotificationService;

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({
        canActivate: (context) => {
          const req = context.switchToHttp().getRequest();
          req.user = { id: 'some-uuid' };
          return true;
        },
      })
      .compile();

    app = moduleFixture.createNestApplication();
    app.setGlobalPrefix('api');
    notificationService = moduleFixture.get<NotificationService>(NotificationService);

    await app.listen(0);
    serverUrl = await app.getUrl();
  }, 30000);

  afterAll(async () => {
    await app.close();
  });

  it('should verify the notifications contract', async () => {
    const verifier = new Verifier({
      provider: 'api',
      providerBaseUrl: serverUrl,
      pactUrls: [path.resolve(process.cwd(), '../../pacts/web-api.json')],
      stateHandlers: {
        'user has notifications': async () => {
          // Prepare mocking of the service
          const notif = Notification.create({
            userId: 'some-uuid',
            type: NotificationType.SYSTEM,
            title: 'Welcome',
            message: 'Welcome to the app',
            channel: NotificationChannel.IN_APP,
          });

          // Use vi.spyOn if we wanted to mock the service,
          // or just ensure the service returns this if it's using a real DB.
          // Since we want simple contract verification, we can mock the service method:
          vi.spyOn(notificationService, 'getUserNotifications').mockResolvedValue([notif as any]);

          return Promise.resolve('User has notifications state handled');
        },
      },
    });

    await verifier.verifyProvider();
  }, 60000);
});
