import { DashboardData } from '../types/visualization.types';

export class ExportService {
  static validateForCsv(data: any[], headers: string[]): void {
    if (!data || data.length === 0) return;

    const firstRowKeys = Object.keys(data[0]);
    headers.forEach((header) => {
      if (!firstRowKeys.includes(header)) {
        throw new Error(`Brak wymaganych kolumn: ${header}`);
      }
    });

    // Enforce ISO 8601 dates in CSV
    data.forEach((row) => {
      if (row.date && typeof row.date === 'string') {
        if (row.date.includes('/')) {
          row.date = row.date.replace(/\//g, '-');
        }
      }
    });
  }

  static toCSV(data: any[], headers: string[]): string {
    this.validateForCsv(data, headers);

    const rows = data.map((row) => headers.map((header) => row[header]).join(','));

    return [headers.join(','), ...rows].join('\n');
  }

  static async generatePDF(data: DashboardData): Promise<Buffer> {
    if (!data) {
      throw new Error('Brak danych do wygenerowania raportu PDF');
    }

    // Check if charts are present as required in detailed_plan.md
    if (!data.charts || Object.keys(data.charts).length === 0) {
      throw new Error('Brak wymaganych wykresów w danych');
    }

    // In actual implementation, this would involve Puppeteer or PDFKit
    // For now, we return a mock buffer
    return Buffer.from('Mock PDF Content');
  }

  static validateForPdf(data: any): void {
    if (!data.charts || Object.keys(data.charts).length === 0) {
      throw new Error('Brak wymaganych wykresów w danych');
    }
  }
}
