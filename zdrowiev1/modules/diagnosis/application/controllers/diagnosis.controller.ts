import { Controller, Post, Get, Body, Req, UsePipes, Inject, Param } from '@nestjs/common';
import { SymptomCheckerService } from '../services/symptom-checker.service';
import { SymptomReportSchema } from '@monorepo/zod-schemas';
import { ZodValidationPipe } from '../../../../packages/nest-utils/src/index';

@Controller('diagnosis')
export class DiagnosisController {
  constructor(@Inject('SYMPTOM_SERVICE') private readonly service: SymptomCheckerService) {}

  @Post('report')
  @UsePipes(
    new ZodValidationPipe(SymptomReportSchema.omit({ id: true, userId: true, createdAt: true })),
  )
  async reportSymptoms(@Body() validated: any, @Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.reportSymptoms(userId, validated.symptoms);
  }

  @Get('history')
  async getHistory(@Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.getUserReports(userId);
  }

  @Get('report/:id/triage')
  async getTriage(@Param('id') id: string) {
    return this.service.getReportTriage(id);
  }
}
