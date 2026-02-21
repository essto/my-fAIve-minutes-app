import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

export const createHttpClient = (config?: AxiosRequestConfig): AxiosInstance => {
    const client = axios.create({
        timeout: 10000,
        headers: {
            'Content-Type': 'application/json',
        },
        ...config,
    });

    // Request Interceptor for Auth
    client.interceptors.request.use(
        (config) => {
            const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;
            if (token && config.headers) {
                config.headers.Authorization = `Bearer ${token}`;
            }
            return config;
        },
        (error) => Promise.reject(error),
    );

    // Response Interceptor for Error Handling
    client.interceptors.response.use(
        (response) => response,
        (error) => {
            // Handle generic errors (e.g., 401, 500)
            if (error.response?.status === 401) {
                // Handle unauthorized (e.g., redirect to login)
            }
            return Promise.reject(error);
        },
    );

    return client;
};
