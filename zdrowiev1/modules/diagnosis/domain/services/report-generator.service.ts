import PDFDocument from 'pdfkit';

export class ReportGeneratorService {
  private readonly disclaimers = {
    pl: 'UWAGA: Ten raport został wygenerowany automatycznie i nie zastępuje profesjonalnej porady lekarskiej. W sytuacjach nagłych dwoń 112.',
    en: 'DISCLAIMER: This report is auto-generated and does not replace professional medical advice. In case of emergency, call 911/112.',
  };

  async generatePdf(data: {
    userId: string;
    reportId?: string;
    symptoms: any[];
    triageResult: string;
    healthHistory?: any[];
    locale?: 'pl' | 'en';
  }): Promise<Buffer> {
    if (!data.userId) {
      throw new Error('InvalidReportDataError');
    }

    const locale = data.locale || 'pl';

    return new Promise((resolve, reject) => {
      const doc = new PDFDocument({ margin: 50 });
      const chunks: any[] = [];

      doc.on('data', (chunk) => chunks.push(chunk));
      doc.on('end', () => resolve(Buffer.concat(chunks)));
      doc.on('error', reject);

      // --- Header & Branding ---
      doc.rect(0, 0, 612, 50).fill('#1A237E'); // Navy Blue Header
      doc.fillColor('#FFFFFF').fontSize(20).text('Zdrowie App', 50, 15);
      doc.fillColor('#000000').fontSize(25).text('Raport Diagnostyczny', 50, 80);

      // Personalization
      doc.fontSize(10).text(`ID: ${data.reportId || 'N/A'}`, 450, 85);
      doc.fontSize(12).text(`Pacjent: ${data.userId}`, 50, 120);
      doc
        .fillColor(data.triageResult === 'HIGH' ? '#D32F2F' : '#2E7D32')
        .text(`Wynik Triage: ${data.triageResult}`, 50, 140);
      doc.fillColor('#000000');

      // --- Symptoms Section ---
      doc.moveDown().fontSize(16).text('Zgłoszone Objawy', { underline: true });
      data.symptoms.forEach((s, i) => {
        doc
          .fontSize(12)
          .text(`${i + 1}. ${s.name || s.nazwa} (Nasielenie: ${s.severity || '?'}/10)`, 70);
      });

      // --- Medical History Section ---
      if (data.healthHistory && data.healthHistory.length > 0) {
        doc.moveDown().fontSize(16).text('Historia Medyczna', { underline: true });
        data.healthHistory.forEach((h, i) => {
          doc.fontSize(12).text(`• [${h.date}] ${h.type}: ${h.value}`, 70);
        });
      }

      // --- Footer & Disclaimer ---
      const disclaimerText = this.disclaimers[locale];
      doc
        .fontSize(8)
        .fillColor('#757575')
        .text(disclaimerText, 50, 720, { width: 500, align: 'center' });

      doc.end();
    });
  }
}
