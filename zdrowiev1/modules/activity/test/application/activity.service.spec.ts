import { describe, it, expect, vi, beforeEach, Mocked } from 'vitest';
import { ActivityService } from '../../src/application/activity.service';
import { IActivityRepository } from '../../src/domain/ports/activity.repository';
import { ActivityEntry } from '@monorepo/shared-types';

describe('ActivityService', () => {
  let service: ActivityService;
  let repository: Mocked<IActivityRepository>;

  beforeEach(() => {
    repository = {
      save: vi.fn(),
      findByUserIdAndDate: vi.fn(),
      getDailySummary: vi.fn(),
    } as any;

    service = new ActivityService(repository);
  });

  describe('logActivity', () => {
    it('should log a new activity and return the created entry (RED)', async () => {
      const userId = 'user-1';
      const command = {
        date: '2026-02-22',
        steps: 1000,
        caloriesBurned: 50,
        activityType: 'WALKING',
        durationMinutes: 15,
      };

      const mockEntry: ActivityEntry = {
        id: 'activity-1',
        userId,
        ...command,
        createdAt: new Date(),
      };

      repository.save.mockResolvedValue(mockEntry);

      const result = await service.logActivity(userId, command);

      expect(result).toEqual(mockEntry);
      expect(repository.save).toHaveBeenCalled();
    });
  });

  describe('getDailySummary', () => {
    it('should return a daily summary for a given date (RED)', async () => {
      const userId = 'user-1';
      const date = '2026-02-22';
      const mockSummary: ActivityEntry = {
        id: 'activity-1',
        userId,
        date,
        steps: 5000,
        caloriesBurned: 250,
        createdAt: new Date(),
      };

      repository.getDailySummary.mockResolvedValue(mockSummary);

      const result = await service.getDailySummary(userId, date);

      expect(result).toEqual(mockSummary);
      expect(repository.getDailySummary).toHaveBeenCalledWith(userId, date);
    });
  });
});
