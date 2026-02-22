import { Injectable } from '@nestjs/common';
import { Notification, NotificationType, NotificationChannel } from './notification.entity';
import { NotificationRepository } from '../ports/notification.repository.port';

@Injectable()
export class NotificationService {
  constructor(private readonly repository: NotificationRepository) {}

  async notify(props: {
    userId: string;
    type: NotificationType;
    title: string;
    message: string;
    channel: NotificationChannel;
  }): Promise<Notification> {
    const notification = Notification.create(props);
    return this.repository.create(notification);
  }

  async markAsRead(id: string): Promise<Notification> {
    const notification = await this.repository.findById(id);
    if (!notification) {
      throw new Error(`Notification with id ${id} not found`);
    }

    const updatedNotification = notification.markAsRead();
    return this.repository.update(updatedNotification);
  }

  async getUserNotifications(userId: string): Promise<Notification[]> {
    return this.repository.findByUserId(userId);
  }
}
