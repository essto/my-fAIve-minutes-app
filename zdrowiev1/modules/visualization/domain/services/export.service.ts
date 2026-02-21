export class ExportService {
  static validateForCsv(data: any[], headers: string[]): void {
    if (data.length === 0) return;

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

  static validateForPdf(data: any): void {
    if (!data.charts || data.charts.length === 0) {
      throw new Error('Brak wymaganych wykresów w danych');
    }
  }
}
