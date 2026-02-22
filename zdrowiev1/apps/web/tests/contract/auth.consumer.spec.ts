import { expect, it, describe } from 'vitest';
import { provider } from './pact.setup';
import { MatchersV3 } from '@pact-foundation/pact';
import axios from 'axios';

const { like } = MatchersV3;

describe('Auth API Contract', () => {
    it('powinien wygenerować kontrakt dla logowania', async () => {
        provider
            .given('użytkownik demo istnieje')
            .uponReceiving('poprawne żądanie logowania')
            .withRequest({
                method: 'POST',
                path: '/api/auth/login',
                headers: { 'Content-Type': 'application/json' },
                body: {
                    email: 'demo@example.com',
                    password: 'Password123!'
                }
            })
            .willRespondWith({
                status: 200,
                headers: { 'Content-Type': 'application/json; charset=utf-8' },
                body: like({
                    access_token: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
                })
            });

        await provider.executeTest(async (mockServer) => {
            const response = await axios.post(`${mockServer.url}/api/auth/login`, {
                email: 'demo@example.com',
                password: 'Password123!'
            });

            expect(response.status).toBe(200);
            expect(response.data).toHaveProperty('access_token');
        });
    });
});
