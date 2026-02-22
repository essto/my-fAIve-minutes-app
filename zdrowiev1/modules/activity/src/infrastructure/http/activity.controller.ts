import { Controller, Post, Get, Body, Req, Param, UsePipes, Inject } from '@nestjs/common';
import { ActivityService } from '../../application/activity.service';
import { CreateActivityCommandSchema, CreateActivityCommand } from '../../domain/activity.schema';
import { ZodValidationPipe } from '../../../../../packages/nest-utils/src/zod-validation.pipe';

@Controller('activity')
export class ActivityController {
  constructor(
    @Inject(ActivityService)
    private readonly service: ActivityService,
  ) {}

  @Post()
  @UsePipes(new ZodValidationPipe(CreateActivityCommandSchema))
  async logActivity(@Body() command: CreateActivityCommand, @Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.logActivity(userId, command);
  }

  @Get('summary/:date')
  async getDailySummary(@Param('date') date: string, @Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.getDailySummary(userId, date);
  }
}
