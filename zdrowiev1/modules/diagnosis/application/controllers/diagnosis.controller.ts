import { Controller, Post, Get, Body, Req, UsePipes, Inject } from '@nestjs/common';
import { DiagnosisService } from '../../domain/services/diagnosis.service';
import { SymptomReportSchema } from '@monorepo/zod-schemas';
import { ZodValidationPipe } from '../../../../packages/nest-utils/src/index';

@Controller('diagnosis')
export class DiagnosisController {
  constructor(@Inject('DIAGNOSIS_SERVICE') private readonly service: DiagnosisService) {}

  @Post('report')
  @UsePipes(
    new ZodValidationPipe(SymptomReportSchema.omit({ id: true, userId: true, timestamp: true })),
  )
  async reportSymptoms(@Body() validated: any, @Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.reportSymptoms(userId, validated);
  }

  @Get('history')
  async getHistory(@Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.getHistory(userId);
  }
}
