import { eq, and } from 'drizzle-orm';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { Inject, Injectable } from '@nestjs/common';
import { DATABASE_CONNECTION } from '@monorepo/database';
import { Notification, NotificationType, NotificationChannel } from '../domain/notification.entity';
import { NotificationRepository } from '../ports/notification.repository.port';
import * as schema from '@monorepo/database';

@Injectable()
export class DrizzleNotificationRepository implements NotificationRepository {
  constructor(
    @Inject(DATABASE_CONNECTION)
    private readonly db: NodePgDatabase<typeof schema>,
  ) {}

  async create(notification: Notification): Promise<Notification> {
    const [record] = await this.db
      .insert(schema.notifications)
      .values({
        userId: notification.userId,
        type: notification.type,
        title: notification.title,
        message: notification.message,
        channel: notification.channel,
        isRead: notification.isRead,
        createdAt: notification.createdAt,
        readAt: notification.readAt,
      })
      .returning();

    return this.mapToDomain(record);
  }

  async findById(id: string): Promise<Notification | undefined> {
    const [record] = await this.db
      .select()
      .from(schema.notifications)
      .where(eq(schema.notifications.id, id));
    if (!record) return undefined;
    return this.mapToDomain(record);
  }

  async findByUserId(userId: string): Promise<Notification[]> {
    const records = await this.db
      .select()
      .from(schema.notifications)
      .where(eq(schema.notifications.userId, userId));
    return records.map(this.mapToDomain);
  }

  async update(notification: Notification): Promise<Notification> {
    if (!notification.id) throw new Error('ID is required for update');

    const [record] = await this.db
      .update(schema.notifications)
      .set({
        isRead: notification.isRead,
        readAt: notification.readAt,
      })
      .where(eq(schema.notifications.id, notification.id))
      .returning();

    return this.mapToDomain(record);
  }

  private mapToDomain(record: any): Notification {
    return new Notification(
      record.id,
      record.userId,
      record.type as NotificationType,
      record.title,
      record.message,
      record.channel as NotificationChannel,
      record.isRead,
      record.createdAt,
      record.readAt || undefined,
    );
  }
}
