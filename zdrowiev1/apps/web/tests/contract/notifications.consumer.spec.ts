import { expect, it, describe } from 'vitest';
import { provider } from './pact.setup';
import { MatchersV3 } from '@pact-foundation/pact';
import axios from 'axios';

const { eachLike, like } = MatchersV3;

describe('Notifications API Contract', () => {
    it('should generate contract for getting notifications', async () => {
        provider
            .given('user has notifications')
            .uponReceiving('a request for all notifications')
            .withRequest({
                method: 'GET',
                path: '/api/notifications',
                headers: { 'Authorization': 'Bearer test-token' }
            })
            .willRespondWith({
                status: 200,
                headers: { 'Content-Type': 'application/json; charset=utf-8' },
                body: eachLike({
                    id: like('some-uuid'),
                    type: like('SYSTEM'),
                    title: like('Welcome'),
                    message: like('Welcome to the app'),
                    channel: like('IN_APP'),
                    isRead: like(false),
                    createdAt: like('2023-01-01T00:00:00.000Z')
                })
            });

        await provider.executeTest(async (mockServer) => {
            const response = await axios.get(`${mockServer.url}/api/notifications`, {
                headers: { 'Authorization': 'Bearer test-token' }
            });

            expect(response.status).toBe(200);
            expect(Array.isArray(response.data)).toBe(true);
        });
    });
});
