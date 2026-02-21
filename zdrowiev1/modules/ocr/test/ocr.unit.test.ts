import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ScannerService } from '../domain/services/scanner.service';
import { ParserService } from '../domain/services/parser.service';
import { OcrValidationService } from '../domain/services/ocr-validation.service';

describe('OCR Module (Etap 5) - Unit Tests', () => {
  describe('ScannerService', () => {
    let scanner: ScannerService;

    beforeEach(() => {
      scanner = new ScannerService();
    });

    it('TC1.1: should reject unsupported file format', async () => {
      await expect(scanner.validateFile('raport.txt')).rejects.toThrow(
        'UnsupportedFileTypeException',
      );
    });

    it('TC1.2: should accept valid PDF format', async () => {
      const result = await scanner.validateFile('wyniki.pdf');
      expect(result).toBe(true);
    });

    it('TC1.3: should return image (bypass) for deskewing for now', async () => {
      const image = { data: 'dummy' };
      const result = await scanner.deskewImage(image);
      expect(result).toBe(image);
    });
  });

  describe('ParserService', () => {
    let parser: ParserService;

    beforeEach(() => {
      parser = new ParserService();
    });

    it('TC2.1: should automatically redact patient PII', async () => {
      const rawText = 'Pacjent: Jan Kowalski PESEL: 99011212345';
      const result = await parser.anonymizeData(rawText);
      expect(result).toBe('Pacjent: [REDACTED] PESEL: [REDACTED]');
    });

    it('TC2.2: should map incomplete data from OCR', async () => {
      const ocrResult = { text: 'Hemoglobina: 14.1 g/dl', confidence: 0.9 };
      const mapped = await parser.mapToMedicalSchema(ocrResult);
      expect(mapped.results[0].referenceRange).toBeNull();
    });

    it('TC2.3: should handle invalid JSON from LLM', async () => {
      await expect(parser.parseWithLLM('{INVALID_JSON')).rejects.toThrow('LLMProcessingException');
    });
  });

  describe('OcrValidationService', () => {
    let validator: OcrValidationService;

    beforeEach(() => {
      validator = new OcrValidationService();
    });

    it('TC3.1: should automatically flag low confidence', async () => {
      const results = [{ parameter: 'HDL', value: '52', confidence: 0.75 }];
      const flagged = await validator.flagLowConfidence(results);
      expect(flagged[0].status).toBe('FLAGGED_FOR_REVIEW');
    });

    it('TC3.2: should ignore high confidence', async () => {
      const results = [{ parameter: 'Glukoza', value: '99', confidence: 0.95 }];
      const verified = await validator.flagLowConfidence(results);
      expect(verified[0].status).toBe('AUTO_VERIFIED');
    });

    it('TC3.3: should allow manual override', async () => {
      const original = { parameter: 'TSH', value: '4.2', confidence: 0.7 };
      const corrected = await validator.manualOverride(original, '3.8');
      expect(corrected.value).toBe('3.8');
      expect(corrected.status).toBe('MANUALLY_VERIFIED');
    });
  });
});
