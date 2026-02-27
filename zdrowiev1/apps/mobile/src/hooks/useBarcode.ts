import { useState } from 'react';
import api from '../services/api';

export interface ScannedProduct {
  name: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

export const useBarcode = () => {
  const [isScanning, setIsScanning] = useState(false);
  const [product, setProduct] = useState<ScannedProduct | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const scanBarcode = () => {
    setIsScanning(true);
    setProduct(null);
    setError(null);
  };

  const cancelScan = () => {
    setIsScanning(false);
    setProduct(null);
    setError(null);
  };

  const lookupProduct = async (barcode: string) => {
    setIsLoading(true);
    setError(null);
    setProduct(null);
    try {
      const response = await api.get(`/diet/food/barcode/${barcode}`);
      setProduct(response.data);
      setIsScanning(false);
    } catch (err: any) {
      if (err.response?.status === 404) {
        setError('Produkt nie znaleziony');
      } else {
        setError(err.message || 'Network Error');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return {
    isScanning,
    product,
    isLoading,
    error,
    scanBarcode,
    cancelScan,
    lookupProduct,
  };
};
