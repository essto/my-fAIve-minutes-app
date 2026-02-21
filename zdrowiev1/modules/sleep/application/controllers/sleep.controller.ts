import { Controller, Post, Get, Body, Req, UsePipes, Inject } from '@nestjs/common';
import { SleepService } from '../../domain/services/sleep.service';
import { SleepRecordSchema } from '@monorepo/zod-schemas';
import { ZodValidationPipe } from '../../../../packages/nest-utils/src/index';

@Controller('sleep')
export class SleepController {
  constructor(@Inject('SLEEP_SERVICE') private readonly service: SleepService) {}

  @Post()
  @UsePipes(
    new ZodValidationPipe(SleepRecordSchema.omit({ id: true, userId: true, createdAt: true })),
  )
  async addRecord(@Body() validated: any, @Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.addRecord(userId, validated);
  }

  @Get()
  async getHistory(@Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.getHistory(userId);
  }

  @Get('efficiency')
  async getEfficiency(@Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    const history = await this.service.getHistory(userId);
    return { efficiency: this.service.calculateEfficiency(history) };
  }
}
