import { useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import { storage } from '../services/storage';

export interface User {
  id: string;
  email: string;
}

export const useAuth = () => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState<string | null>(null);

  const restoreSession = useCallback(async () => {
    setIsLoading(true);
    try {
      const token = await storage.getToken();
      if (!token) {
        setIsAuthenticated(false);
        setUser(null);
        return;
      }

      // Token exists, verify it and fetch user data
      const response = await api.get('/users/me');
      setUser(response.data);
      setIsAuthenticated(true);
    } catch (err: any) {
      if (err.response?.status === 401) {
        await storage.removeToken();
      }
      setIsAuthenticated(false);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    restoreSession();
  }, [restoreSession]);

  const login = async (email: string, password: string) => {
    setError(null);
    try {
      const response = await api.post('/auth/login', { email, password });
      const { access_token, user: userData } = response.data;

      await storage.setToken(access_token);
      setUser(userData);
      setIsAuthenticated(true);
    } catch (err: any) {
      if (err.response?.status === 401) {
        setError(err.response.data?.message || 'Invalid credentials');
      } else {
        setError(err.message || 'Network Error');
      }
      setIsAuthenticated(false);
      setUser(null);
    }
  };

  const logout = async () => {
    await storage.removeToken();
    setIsAuthenticated(false);
    setUser(null);
    setError(null);
  };

  return {
    isAuthenticated,
    isLoading,
    user,
    error,
    login,
    logout,
    restoreSession,
  };
};
