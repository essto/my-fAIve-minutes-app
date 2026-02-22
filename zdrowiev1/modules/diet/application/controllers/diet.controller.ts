import { Controller, Post, Get, Body, Req, Query, UsePipes, Inject, Param } from '@nestjs/common';
import { MealService } from '../services/meal.service';
import { BarcodeLookupService } from '../../domain/services/barcode-lookup.service';
import { MealSchema } from '@monorepo/zod-schemas';
import { ZodValidationPipe } from '../../../../packages/nest-utils/src/index';

@Controller('diet')
export class DietController {
  constructor(
    @Inject('MEAL_SERVICE') private readonly service: MealService,
    @Inject('BARCODE_SERVICE') private readonly barcodeService: BarcodeLookupService,
  ) {}

  @Post()
  @UsePipes(new ZodValidationPipe(MealSchema.omit({ id: true, userId: true, consumedAt: true })))
  async logMeal(@Body() validated: any, @Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.logMeal(userId, validated.name, validated.products);
  }

  @Get('summary')
  async getSummary(@Req() req: any, @Query('date') dateParam?: string) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    const date = dateParam ? new Date(dateParam) : new Date();
    return this.service.getDailySummary(userId, date);
  }

  @Get('history')
  async getHistory(@Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.getDailySummary(userId); // Simplified for now
  }

  @Get('barcode/:code')
  async lookupBarcode(@Param('code') code: string) {
    return this.barcodeService.lookupBarcode(code);
  }
}
