import { describe, it, expect, beforeEach } from 'vitest';
import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import { createHttpClient } from '../src/index';

describe('api-client', () => {
    let mock: MockAdapter;

    beforeEach(() => {
        mock = new MockAdapter(axios);
    });

    it('adds auth token if present in localStorage', async () => {
        // Mock window and localStorage
        global.window = {} as any;
        global.localStorage = {
            getItem: () => 'test-token',
        } as any;

        const client = createHttpClient({ baseURL: 'http://test.com' });
        mock.onGet('/test').reply(200);

        const response = await client.get('/test');
        expect(response.config.headers?.Authorization).toBe('Bearer test-token');

        // Clean up
        delete (global as any).window;
        delete (global as any).localStorage;
    });

    it('handles 401 errors', async () => {
        const client = createHttpClient({ baseURL: 'http://test.com' });
        mock.onGet('/fail').reply(401);

        await expect(client.get('/fail')).rejects.toThrow();
    });
});
