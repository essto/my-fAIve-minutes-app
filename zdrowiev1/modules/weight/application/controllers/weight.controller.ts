import { Controller, Post, Get, Body, Req, UsePipes, Inject } from '@nestjs/common';
import { WeightService } from '../../domain/services/weight.service';
import { WeightReadingSchema } from '@monorepo/zod-schemas';
import { ZodValidationPipe } from '../../../../packages/nest-utils/src/index';

@Controller('weight')
export class WeightController {
  constructor(@Inject('WEIGHT_SERVICE') private readonly service: WeightService) {}

  @Post()
  @UsePipes(
    new ZodValidationPipe(WeightReadingSchema.omit({ id: true, userId: true, timestamp: true })),
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

  @Post('bmi')
  async calculateBMI(@Body() body: { weight: number; height: number }) {
    return { bmi: this.service.calculateBMI(body.weight, body.height) };
  }
}
