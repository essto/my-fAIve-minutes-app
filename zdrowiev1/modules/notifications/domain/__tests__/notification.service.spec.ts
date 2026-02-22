import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NotificationService } from '../notification.service';
import { NotificationRepository } from '../../ports/notification.repository.port';
import { Notification, NotificationType, NotificationChannel } from '../notification.entity';

describe('NotificationService', () => {
  let service: NotificationService;
  let mockRepo: NotificationRepository;

  beforeEach(() => {
    mockRepo = {
      create: vi.fn(),
      findById: vi.fn(),
      findByUserId: vi.fn(),
      update: vi.fn(),
    } as unknown as NotificationRepository;
    service = new NotificationService(mockRepo);
  });

  it('should create and persist a notification', async () => {
    const props = {
      userId: 'user-1',
      type: NotificationType.HEALTH_ALERT,
      title: 'High HR',
      message: 'Your heart rate is high',
      channel: NotificationChannel.IN_APP,
    };

    const notification = Notification.create(props);
    vi.mocked(mockRepo.create).mockResolvedValue({ ...notification, id: 'notif-1' } as any);

    const result = await service.notify(props);

    expect(mockRepo.create).toHaveBeenCalled();
    expect(result.id).toBe('notif-1');
    expect(result.userId).toBe(props.userId);
  });

  it('should mark a notification as read', async () => {
    const notification = new Notification(
      'notif-1',
      'user-1',
      NotificationType.SYSTEM,
      'Welcome',
      'Welcome to the app',
      NotificationChannel.IN_APP,
      false,
      new Date(),
    );

    vi.mocked(mockRepo.findById).mockResolvedValue(notification);
    vi.mocked(mockRepo.update).mockImplementation(async (n) => n);

    const result = await service.markAsRead('notif-1');

    expect(result.isRead).toBe(true);
    expect(result.readAt).toBeDefined();
    expect(mockRepo.update).toHaveBeenCalledWith(expect.objectContaining({ isRead: true }));
  });
});
