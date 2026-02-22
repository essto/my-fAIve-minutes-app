import { Injectable, Inject } from '@nestjs/common';
import { IActivityRepository } from '../domain/ports/activity.repository';
import { ActivityEntry } from '@monorepo/shared-types';
import { CreateActivityCommand } from '../domain/activity.schema';

@Injectable()
export class ActivityService {
  constructor(
    @Inject('IActivityRepository')
    private readonly repository: IActivityRepository,
  ) {}

  async logActivity(userId: string, command: CreateActivityCommand): Promise<ActivityEntry> {
    const entry: Partial<ActivityEntry> = {
      userId,
      ...command,
    };
    return this.repository.save(entry);
  }

  async getDailySummary(userId: string, date: string): Promise<ActivityEntry | null> {
    return this.repository.getDailySummary(userId, date);
  }
}
