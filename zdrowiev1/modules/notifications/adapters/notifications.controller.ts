import {
  Controller,
  Get,
  Param,
  Patch,
  UseGuards,
  Request as NestRequest,
  Inject,
} from '@nestjs/common';
import { NotificationService } from '../domain/notification.service';
import { JwtAuthGuard } from '@monorepo/auth';

@Controller('notifications')
export class NotificationController {
  constructor(
    @Inject(NotificationService)
    private readonly notificationService: NotificationService,
  ) {}

  @UseGuards(JwtAuthGuard)
  @Get()
  async getNotifications(@NestRequest() req: any) {
    const userId = req.user.id;
    return this.notificationService.getUserNotifications(userId);
  }

  @UseGuards(JwtAuthGuard)
  @Patch(':id/read')
  async markAsRead(@Param('id') id: string) {
    return this.notificationService.markAsRead(id);
  }
}
