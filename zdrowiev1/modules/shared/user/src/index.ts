import { db, users } from '@monorepo/database';
import { eq } from 'drizzle-orm';
import { User } from '@monorepo/shared-types';

export const UserService = {
    async findById(id: string) {
        const [user] = await db.select().from(users).where(eq(users.id, id));
        return user;
    },

    async findByEmail(email: string) {
        const [user] = await db.select().from(users).where(eq(users.email, email));
        return user;
    },

    async create(data: Partial<User>) {
        const [user] = await db.insert(users).values(data as any).returning();
        return user;
    },
};
