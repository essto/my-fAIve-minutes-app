import { Notification } from '../domain/notification.entity';

export interface NotificationSender {
  send(notification: Notification): Promise<void>;
}

export const NotificationSenderToken = Symbol('NotificationSender');

export class InAppNotificationSender implements NotificationSender {
  async send(notification: Notification): Promise<void> {
    // In a real app, this might emit a WebSocket event or update a Redis stream
    console.log(`[In-App Notification] Sent to ${notification.userId}: ${notification.title}`);
    return Promise.resolve();
  }
}
