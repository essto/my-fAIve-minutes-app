import { NotificationService } from './notification.service';
import { NotificationType, NotificationChannel } from './notification.entity';

export class AnomalyDetectorService {
  constructor(private readonly notificationService: NotificationService) {}

  async detectHeartRateAnomaly(data: {
    userId: string;
    value: number;
    timestamp: Date;
  }): Promise<void> {
    if (data.value > 100) {
      await this.notificationService.notify({
        userId: data.userId,
        type: NotificationType.HEALTH_ALERT,
        title: 'High Heart Rate Alert',
        message: `High heart rate detected: ${data.value} bpm`,
        channel: NotificationChannel.IN_APP,
      });
    } else if (data.value < 40) {
      await this.notificationService.notify({
        userId: data.userId,
        type: NotificationType.HEALTH_ALERT,
        title: 'Low Heart Rate Alert',
        message: `Low heart rate detected: ${data.value} bpm`,
        channel: NotificationChannel.IN_APP,
      });
    }
  }

  async detectSleepAnomaly(data: { userId: string; duration: number; date: Date }): Promise<void> {
    if (data.duration < 4) {
      await this.notificationService.notify({
        userId: data.userId,
        type: NotificationType.HEALTH_ALERT,
        title: 'Low Sleep Duration',
        message: `Low sleep duration: ${data.duration} hours`,
        channel: NotificationChannel.IN_APP,
      });
    }
  }
}
