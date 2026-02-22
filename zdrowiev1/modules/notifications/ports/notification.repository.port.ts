import { Notification } from '../domain/notification.entity';

export interface NotificationRepository {
  create(notification: Notification): Promise<Notification>;
  findById(id: string): Promise<Notification | undefined>;
  findByUserId(userId: string): Promise<Notification[]>;
  update(notification: Notification): Promise<Notification>;
}

export const NotificationRepositoryToken = Symbol('NotificationRepository');
