import { Controller, Post, Get, Body, Req, Query, UsePipes, Inject, Param } from '@nestjs/common';
import { DietService } from '../../domain/services/diet.service';
import { BarcodeLookupService } from '../../domain/services/barcode-lookup.service';
import { MealEntrySchema } from '@monorepo/zod-schemas';
import { ZodValidationPipe } from '../../../../packages/nest-utils/src/index';

@Controller('diet')
export class DietController {
  constructor(
    @Inject('DIET_SERVICE') private readonly service: DietService,
    @Inject('BARCODE_SERVICE') private readonly barcodeService: BarcodeLookupService,
  ) {}

  @Post()
  @UsePipes(
    new ZodValidationPipe(MealEntrySchema.omit({ id: true, userId: true, timestamp: true })),
  )
  async addEntry(@Body() validated: any, @Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.addEntry(userId, validated);
  }

  @Get()
  async getHistory(@Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.getHistory(userId);
  }

  @Get('analysis')
  async getAnalysis(@Req() req: any, @Query('date') dateParam?: string) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    const date = dateParam ? new Date(dateParam) : new Date();
    // Default targets (this could be fetched from User settings in future)
    const targets = { calories: 2000, protein: 70, carbs: 250, fat: 70 };
    return this.service.analyzeDailyNutrition(userId, date, targets);
  }

  @Get('barcode/:code')
  async lookupBarcode(@Param('code') code: string) {
    return this.barcodeService.lookupBarcode(code);
  }
}
