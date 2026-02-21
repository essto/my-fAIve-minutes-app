import { Module } from '@nestjs/common';
import { OcrController } from './application/controllers/ocr.controller';
import { OcrOrchestrationService } from './domain/services/ocr-orchestration.service';
import { ScannerService } from './domain/services/scanner.service';
import { ParserService } from './domain/services/parser.service';
import { OcrValidationService } from './domain/services/ocr-validation.service';

@Module({
  controllers: [OcrController],
  providers: [ScannerService, ParserService, OcrValidationService, OcrOrchestrationService],
  exports: [OcrOrchestrationService],
})
export class OcrModule {}
