import { useState } from 'react';
import { z } from 'zod';
import api from '../services/api';

const ProductSchema = z.object({
  name: z.string().min(1),
  calories: z.number().min(0),
  protein: z.number().min(0),
  carbs: z.number().min(0),
  fat: z.number().min(0),
  quantity: z.number().min(1),
});

export const MealLogSchema = z.object({
  name: z.string().min(1),
  products: z.array(ProductSchema).min(1),
});

export type Product = z.infer<typeof ProductSchema>;
export type MealLog = z.infer<typeof MealLogSchema>;

export interface Meal {
  id: string;
  name: string;
  consumedAt: string;
  products: Product[];
}

export interface DailySummary {
  total: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
  isDeficit: boolean;
  isSurplus: boolean;
}

export const useDietData = () => {
  const [meals, setMeals] = useState<Meal[]>([]);
  const [dailySummary, setDailySummary] = useState<DailySummary | null>(null);

  const [barcodeResult, setBarcodeResult] = useState<Product | null>(null);
  const [barcodeError, setBarcodeError] = useState<string | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<z.ZodError | null>(null);

  const fetchMeals = async (date: string) => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await api.get(`/diet/meals?date=${date}`);
      setMeals(response.data);
    } catch (err: any) {
      setError(err.message || 'Wystąpił błąd podczas pobierania posiłków');
    } finally {
      setIsLoading(false);
    }
  };

  const logMeal = async (data: MealLog) => {
    try {
      setIsLoading(true);
      setError(null);
      setValidationError(null);

      // Walidacja Zod przed POST
      const parsed = MealLogSchema.safeParse(data);
      if (!parsed.success) {
        setValidationError(parsed.error);
        setIsLoading(false);
        return;
      }

      await api.post('/diet/meals', parsed.data);
      // Oczekujemy, że użytkownik fetchuje posiłki po zalogowaniu nowego
    } catch (err: any) {
      setError(err.message || 'Wystąpił błąd podczas logowania posiłku');
    } finally {
      setIsLoading(false);
    }
  };

  const lookupBarcode = async (barcode: string) => {
    try {
      setIsLoading(true);
      setBarcodeResult(null);
      setBarcodeError(null);

      const response = await api.get(`/diet/products/barcode/${barcode}`);
      setBarcodeResult(response.data);
      return response.data;
    } catch (err: any) {
      setBarcodeError('Produkt nie znaleziony');
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const fetchDailySummary = async (date: string) => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await api.get(`/diet/summary?date=${date}`);
      setDailySummary(response.data);
    } catch (err: any) {
      setError(err.message || 'Wystąpił błąd podczas pobierania podsumowania');
    } finally {
      setIsLoading(false);
    }
  };

  return {
    meals,
    dailySummary,
    barcodeResult,
    barcodeError,
    isLoading,
    error,
    validationError,
    fetchMeals,
    logMeal,
    lookupBarcode,
    fetchDailySummary,
  };
};
