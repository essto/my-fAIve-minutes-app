import { useState } from 'react';
import * as ImagePicker from 'expo-image-picker';
import api from '../services/api';

export interface OCRResult {
  text: string;
  confidence: number;
}

export const useOCR = () => {
  const [ocrResult, setOcrResult] = useState<OCRResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const processImage = async (uri: string) => {
    try {
      setIsLoading(true);
      setError(null);
      setOcrResult(null);

      // In a real app, we would use FormData to upload the file to Next.js API
      // Since fetch/axios in RN needs specific configuration for binary uploads,
      // we simulate the API call based on the TDD spec.

      const formData = new FormData();
      formData.append('file', {
        uri,
        name: 'scan.jpg',
        type: 'image/jpeg',
      } as any);

      const response = await api.post('/ocr/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setOcrResult(response.data);
    } catch (err: any) {
      setError(err.message || 'Wystąpił błąd podczas analizy obrazu');
    } finally {
      setIsLoading(false);
    }
  };

  const pickFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        await processImage(result.assets[0].uri);
      }
    } catch (err: any) {
      setError('Nie udało się otworzyć galerii.');
    }
  };

  const takePhoto = async () => {
    try {
      const permissionResult = await ImagePicker.requestCameraPermissionsAsync();

      if (permissionResult.granted === false) {
        setError('Aplikacja wymaga dostępu do aparatu.');
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        await processImage(result.assets[0].uri);
      }
    } catch (err: any) {
      setError('Nie udało się zrobić zdjęcia.');
    }
  };

  const clearResult = () => {
    setOcrResult(null);
    setError(null);
  };

  return {
    ocrResult,
    isLoading,
    error,
    pickFromGallery,
    takePhoto,
    clearResult,
  };
};
