import { Module, Global } from '@nestjs/common';
import { NotificationService } from './domain/notification.service';
import { AnomalyDetectorService } from './domain/anomaly-detector.service';
import { NotificationRepositoryToken } from './ports/notification.repository.port';
import { DrizzleNotificationRepository } from './adapters/drizzle-notification.repository';
import {
  NotificationSenderToken,
  InAppNotificationSender,
} from './adapters/in-app-notification.sender';
import { DatabaseModule } from '@monorepo/shared-database';

@Global()
@Module({
  imports: [DatabaseModule],
  providers: [
    {
      provide: NotificationRepositoryToken,
      useClass: DrizzleNotificationRepository,
    },
    {
      provide: NotificationSenderToken,
      useClass: InAppNotificationSender,
    },
    {
      provide: NotificationService,
      useFactory: (repo) => new NotificationService(repo),
      inject: [NotificationRepositoryToken],
    },
    {
      provide: AnomalyDetectorService,
      useFactory: (notifService) => new AnomalyDetectorService(notifService),
      inject: [NotificationService],
    },
  ],
  exports: [NotificationService, AnomalyDetectorService],
})
export class NotificationsModule {}
