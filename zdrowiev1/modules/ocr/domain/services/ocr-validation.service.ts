export class OcrValidationService {
  async flagLowConfidence(results: any[]): Promise<any[]> {
    return results.map((res) => ({
      ...res,
      status: res.confidence < 0.8 ? 'FLAGGED_FOR_REVIEW' : 'AUTO_VERIFIED',
    }));
  }

  async manualOverride(result: any, newValue: any): Promise<any> {
    return {
      ...result,
      value: newValue,
      status: 'MANUALLY_VERIFIED',
    };
  }
}
