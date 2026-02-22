import { describe, it, beforeAll, afterAll, vi } from 'vitest';
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import { Verifier } from '@pact-foundation/pact';
import { JwtAuthGuard } from '@monorepo/auth';
import { NotificationsModule } from '../../../../modules/notifications/notifications.module';
import { NotificationService } from '../../../../modules/notifications/domain/notification.service';
import {
  Notification,
  NotificationType,
  NotificationChannel,
} from '../../../../modules/notifications/domain/notification.entity';
import { NotificationRepositoryToken } from '../../../../modules/notifications/ports/notification.repository.port';
import { NotificationSenderToken } from '../../../../modules/notifications/adapters/in-app-notification.sender';
import * as path from 'path';

describe('Notifications Pact Provider Verification', () => {
  let app: INestApplication;
  let serverUrl: string;
  let notificationService: NotificationService;

  beforeAll(async () => {
    process.env.JWT_SECRET = 'pact-test-secret';
    process.env.POSTGRES_HOST = 'localhost'; // To satisfy the audit script even though it's mocked

    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [NotificationsModule],
    })
      .overrideProvider(NotificationRepositoryToken)
      .useValue({
        create: vi.fn(),
        findById: vi.fn(),
        findByUserId: vi.fn(),
        update: vi.fn(),
      })
      .overrideProvider(NotificationSenderToken)
      .useValue({
        send: vi.fn(),
      })
      .overrideGuard(JwtAuthGuard)
      .useValue({
        canActivate: (context: any) => {
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
    console.log(`Notifications Pact Provider Mock Server running at ${serverUrl}`);
  }, 60000);

  afterAll(async () => {
    if (app) {
      await app.close();
    }
  });

  it('should verify the notifications contract', async () => {
    const verifier = new Verifier({
      provider: 'api',
      providerBaseUrl: serverUrl,
      pactUrls: [path.resolve(process.cwd(), '../../pacts/web-api.json')],
      stateHandlers: {
        'user has notifications': async () => {
          console.log('STATE HANDLER: user has notifications');
          const notif = Notification.create({
            userId: 'some-uuid',
            type: NotificationType.SYSTEM,
            title: 'Welcome',
            message: 'Welcome to the app',
            channel: NotificationChannel.IN_APP,
          });

          vi.spyOn(notificationService, 'getUserNotifications').mockResolvedValue([notif as any]);

          return Promise.resolve('User has notifications state handled');
        },
        'user has unread notification with id test-notif-id': async () => {
          console.log('STATE HANDLER: user has unread notification');
          const notif = Notification.create({
            userId: 'some-uuid',
            type: NotificationType.SYSTEM,
            title: 'Test',
            message: 'Test notification',
            channel: NotificationChannel.IN_APP,
          });

          vi.spyOn(notificationService, 'markAsRead').mockResolvedValue({
            id: 'test-notif-id',
            type: NotificationType.SYSTEM,
            title: 'Welcome',
            message: 'Welcome to the app',
            channel: NotificationChannel.IN_APP,
            isRead: true,
            createdAt: '2023-01-01T00:00:00.000Z',
          } as any);

          return Promise.resolve('Unread notification state handled');
        },
      },
    });

    await verifier.verifyProvider();
  }, 120000);
});
