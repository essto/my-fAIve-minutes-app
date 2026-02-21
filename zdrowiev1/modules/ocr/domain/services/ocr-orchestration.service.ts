import { Injectable } from '@nestjs/common';
import { ScannerService } from './scanner.service';
import { ParserService } from './parser.service';
import { OcrValidationService } from './ocr-validation.service';

@Injectable()
export class OcrOrchestrationService {
  constructor(
    private readonly scanner: ScannerService,
    private readonly parser: ParserService,
    private readonly validator: OcrValidationService,
  ) {}

  async processDocument(userId: string, filename: string, fileBuffer: Buffer): Promise<any> {
    // 1. Validate
    await this.scanner.validateFile(filename);

    // 2. Preprocess
    const cleanImage = await this.scanner.preprocessImage(fileBuffer);

    // 3. OCR Extraction (Mocked for now as we don't handle real image files in unit logic)
    const rawOcrText = 'Pacjent: Jan Kowalski PESEL: 99011212345 Hemoglobina: 14.1 g/dl';

    // 4. Anonymize
    const anonymizedText = await this.parser.anonymizeData(rawOcrText);

    // 5. LLM Parse
    const structuredData = await this.parser.mapToMedicalSchema({
      text: anonymizedText,
      confidence: 0.85,
    });

    // 6. Validate/Flag
    const validatedResults = await this.validator.flagLowConfidence(structuredData.results);

    return {
      userId,
      documentType: 'Laboratory Results',
      results: validatedResults,
      rawText: anonymizedText,
    };
  }

  async verifyResult(result: any, overrideValue?: string): Promise<any> {
    if (overrideValue) {
      return this.validator.manualOverride(result, overrideValue);
    }
    return { ...result, status: 'AUTO_VERIFIED' };
  }
}
