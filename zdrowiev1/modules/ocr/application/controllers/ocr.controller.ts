import {
  Controller,
  Post,
  Body,
  Req,
  UseInterceptors,
  UploadedFile,
  Get,
  Param,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { OcrOrchestrationService } from '../../domain/services/ocr-orchestration.service';

@Controller('ocr')
export class OcrController {
  constructor(private readonly ocrService: OcrOrchestrationService) {}

  @Post('upload')
  @UseInterceptors(FileInterceptor('file'))
  async uploadDocument(@UploadedFile() file: any, @Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.ocrService.processDocument(userId, file.originalname, file.buffer);
  }

  @Post('verify')
  async verifyResult(@Body() body: { result: any; overrideValue?: string }) {
    return this.ocrService.verifyResult(body.result, body.overrideValue);
  }
}
