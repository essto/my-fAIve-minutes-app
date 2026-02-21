import { Controller, Post, Get, Body, Req, UsePipes, Inject, Res, Param } from '@nestjs/common';
import { Response } from 'express';
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

  @Get('report/:id/pdf')
  async downloadReport(@Param('id') diagnosisId: string, @Req() req: any, @Res() res: Response) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    const buffer = await this.service.generateReport(userId, diagnosisId);

    res.set({
      'Content-Type': 'application/pdf',
      'Content-Disposition': `attachment; filename=diagnosis-report-${diagnosisId}.pdf`,
      'Content-Length': buffer.length,
    });

    res.end(buffer);
  }

  @Get('history')
  async getHistory(@Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.service.getHistory(userId);
  }
}
