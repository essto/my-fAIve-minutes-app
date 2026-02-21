import { Controller, Post, Get, Body, Req, UsePipes, Inject } from '@nestjs/common';
import { HeartRateService } from '../../domain/services/heart-rate.service';
import { HeartRateReadingSchema } from '@monorepo/zod-schemas';
import { ZodValidationPipe } from '../../../../packages/nest-utils/src/index';

@Controller('heart-rate')
export class HeartRateController {
  constructor(@Inject('HEART_RATE_SERVICE') private readonly service: HeartRateService) {}

  @Post()
  @UsePipes(
    new ZodValidationPipe(HeartRateReadingSchema.omit({ id: true, userId: true, timestamp: true })),
  )
  async addReading(@Body() validated: any, @Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.addReading(userId, validated);
  }

  @Get()
  async getHistory(@Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.getHistory(userId);
  }

  @Get('resting-average')
  async getRestingAverage(@Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    const history = await this.service.getHistory(userId);
    return { average: this.service.analyzeRestingHR(history) };
  }
}
