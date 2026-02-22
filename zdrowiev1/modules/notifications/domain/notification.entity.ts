export enum NotificationType {
  SYSTEM = 'SYSTEM',
  HEALTH_ALERT = 'HEALTH_ALERT',
  REMINDER = 'REMINDER',
}

export enum NotificationChannel {
  IN_APP = 'IN_APP',
  EMAIL = 'EMAIL',
  PUSH = 'PUSH',
}

export class Notification {
  constructor(
    public readonly id: string | undefined,
    public readonly userId: string,
    public readonly type: NotificationType,
    public readonly title: string,
    public readonly message: string,
    public readonly channel: NotificationChannel,
    public readonly isRead: boolean = false,
    public readonly createdAt: Date = new Date(),
    public readonly readAt: Date | undefined = undefined,
  ) {}

  static create(props: {
    userId: string;
    type: NotificationType;
    title: string;
    message: string;
    channel: NotificationChannel;
  }): Notification {
    return new Notification(
      undefined,
      props.userId,
      props.type,
      props.title,
      props.message,
      props.channel,
      false,
      new Date(),
      undefined,
    );
  }

  markAsRead(): Notification {
    return new Notification(
      this.id,
      this.userId,
      this.type,
      this.title,
      this.message,
      this.channel,
      true,
      this.createdAt,
      new Date(),
    );
  }
}
