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

    it('should generate contract for marking notification as read', async () => {
        provider
            .given('user has unread notification with id test-notif-id')
            .uponReceiving('a request to mark notification as read')
            .withRequest({
                method: 'PATCH',
                path: '/api/notifications/test-notif-id/read',
                headers: { 'Authorization': 'Bearer test-token' }
            })
            .willRespondWith({
                status: 200,
                headers: { 'Content-Type': 'application/json; charset=utf-8' },
                body: like({
                    id: like('test-notif-id'),
                    type: like('SYSTEM'),
                    title: like('Welcome'),
                    message: like('Welcome to the app'),
                    channel: like('IN_APP'),
                    isRead: true,
                    createdAt: like('2023-01-01T00:00:00.000Z')
                })
            });

        await provider.executeTest(async (mockServer) => {
            const response = await axios.patch(
                `${mockServer.url}/api/notifications/test-notif-id/read`,
                {},
                { headers: { 'Authorization': 'Bearer test-token' } }
            );

            expect(response.status).toBe(200);
            expect(response.data.isRead).toBe(true);
            expect(response.data.id).toBe('test-notif-id');
        });
    });
});
