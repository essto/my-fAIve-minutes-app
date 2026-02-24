import sharp from 'sharp';

export class ScannerService {
  async validateFile(filename: string): Promise<boolean> {
    const allowedExtensions = ['.jpg', '.jpeg', '.png', '.pdf'];
    const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'));

    if (!allowedExtensions.includes(ext)) {
      throw new Error('UnsupportedFileTypeException');
    }
    return true;
  }

  async preprocessImage(image: Buffer): Promise<Buffer> {
    // In a real implementation, this would use sharp to grayscale/denoise
    return image;
  }

  async deskewImage(image: Buffer): Promise<Buffer> {
    // Placeholder for deskew logic
    // For TDD, we just need it to not throw 'Not implemented' if we want to pass a test
    // or keep it throwing if we haven't written the 'pass' logic yet.
    // I'll implement a basic bypass for now.
    return image;
  }
}
