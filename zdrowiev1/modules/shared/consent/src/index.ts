import { db, consents } from '@monorepo/database';
import { eq, and } from 'drizzle-orm';

export const ConsentService = {
    async getByUserId(userId: string) {
        return db.select().from(consents).where(eq(consents.userId, userId));
    },

    async updateConsent(userId: string, category: string, status: 'granted' | 'revoked') {
        const existing = await db
            .select()
            .from(consents)
            .where(and(eq(consents.userId, userId), eq(consents.category, category)));

        if (existing.length > 0) {
            return db
                .update(consents)
                .set({ status, revokedAt: status === 'revoked' ? new Date() : null })
                .where(and(eq(consents.userId, userId), eq(consents.category, category)))
                .returning();
        }

        return db
            .insert(consents)
            .values({ userId, category, status } as any)
            .returning();
    },
};
