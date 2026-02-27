import { useState } from 'react';

export interface SleepData {
  date: string;
  duration: number; // in hours
  quality: number; // 0-100
}

export interface StepsData {
  date: string;
  steps: number;
}

export const useHealthPlatform = () => {
  const [hasPermissions, setHasPermissions] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [sleepData, setSleepData] = useState<SleepData[]>([]);
  const [stepsData, setStepsData] = useState<StepsData[]>([]);
  const [error, setError] = useState<string | null>(null);

  const requestPermissions = async () => {
    setIsLoading(true);
    setError(null);
    try {
      // Mocking native permission request
      await new Promise((resolve) => setTimeout(resolve, 500));
      setHasPermissions(true);
    } catch (err: any) {
      setError(err.message || 'Failed to get permissions');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchSleepData = async (days: number) => {
    if (!hasPermissions) {
      setError('No permission to access health data');
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      // Mocking native fetching
      await new Promise((resolve) => setTimeout(resolve, 500));

      const mockData: SleepData[] = Array.from({ length: days }).map((_, i) => {
        const d = new Date();
        d.setDate(d.getDate() - i);
        return {
          date: d.toISOString().split('T')[0],
          duration: 6 + Math.random() * 3,
          quality: Math.floor(60 + Math.random() * 40),
        };
      });
      setSleepData(mockData);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch sleep data');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchStepsData = async (days: number) => {
    if (!hasPermissions) {
      setError('No permission to access health data');
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      // Mocking native fetching
      await new Promise((resolve) => setTimeout(resolve, 500));

      const mockData: StepsData[] = Array.from({ length: days }).map((_, i) => {
        const d = new Date();
        d.setDate(d.getDate() - i);
        return {
          date: d.toISOString().split('T')[0],
          steps: Math.floor(3000 + Math.random() * 7000),
        };
      });
      setStepsData(mockData);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch steps data');
    } finally {
      setIsLoading(false);
    }
  };

  return {
    hasPermissions,
    isLoading,
    sleepData,
    stepsData,
    error,
    requestPermissions,
    fetchSleepData,
    fetchStepsData,
  };
};
