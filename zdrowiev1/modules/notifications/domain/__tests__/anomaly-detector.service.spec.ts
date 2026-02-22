import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AnomalyDetectorService } from '../anomaly-detector.service';
import { NotificationService } from '../notification.service';
import { NotificationType, NotificationChannel } from '../notification.entity';

describe('AnomalyDetectorService', () => {
  let service: AnomalyDetectorService;
  let mockNotificationService: NotificationService;

  beforeEach(() => {
    mockNotificationService = {
      notify: vi.fn(),
    } as unknown as NotificationService;
    service = new AnomalyDetectorService(mockNotificationService);
  });

  describe('Heart Rate Anomalies', () => {
    it('should detect high heart rate (>100bpm)', async () => {
      const data = { userId: 'user-1', value: 120, timestamp: new Date() };
      await service.detectHeartRateAnomaly(data);

      expect(mockNotificationService.notify).toHaveBeenCalledWith(
        expect.objectContaining({
          type: NotificationType.HEALTH_ALERT,
          title: 'High Heart Rate Alert',
          message: 'High heart rate detected: 120 bpm',
        }),
      );
    });

    it('should detect low heart rate (<40bpm)', async () => {
      const data = { userId: 'user-1', value: 35, timestamp: new Date() };
      await service.detectHeartRateAnomaly(data);

      expect(mockNotificationService.notify).toHaveBeenCalledWith(
        expect.objectContaining({
          type: NotificationType.HEALTH_ALERT,
          title: 'Low Heart Rate Alert',
          message: 'Low heart rate detected: 35 bpm',
        }),
      );
    });

    it('should not detect anomaly for normal heart rate (70bpm)', async () => {
      const data = { userId: 'user-1', value: 70, timestamp: new Date() };
      await service.detectHeartRateAnomaly(data);

      expect(mockNotificationService.notify).not.toHaveBeenCalled();
    });
  });

  describe('Sleep Anomalies', () => {
    it('should detect low sleep duration (<4h)', async () => {
      const data = { userId: 'user-1', duration: 3.5, date: new Date() };
      await service.detectSleepAnomaly(data);

      expect(mockNotificationService.notify).toHaveBeenCalledWith(
        expect.objectContaining({
          type: NotificationType.HEALTH_ALERT,
          title: 'Low Sleep Duration',
          message: 'Low sleep duration: 3.5 hours',
        }),
      );
    });

    it('should not detect anomaly for normal sleep (7h)', async () => {
      const data = { userId: 'user-1', duration: 7, date: new Date() };
      await service.detectSleepAnomaly(data);

      expect(mockNotificationService.notify).not.toHaveBeenCalled();
    });
  });
});
