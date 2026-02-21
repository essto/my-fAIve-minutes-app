import * as bcrypt from 'bcryptjs';
import * as jwt from 'jsonwebtoken';
import { db, users } from '@monorepo/database';
import { eq } from 'drizzle-orm';

const JWT_SECRET = process.env.JWT_SECRET || 'super-secret-key';

export const AuthService = {
    async hashPassword(password: string): Promise<string> {
        return bcrypt.hash(password, 12);
    },

    async comparePassword(password: string, hash: string): Promise<boolean> {
        return bcrypt.compare(password, hash);
    },

    generateToken(userId: string): string {
        return jwt.sign({ userId }, JWT_SECRET, { expiresIn: '1h' });
    },

    verifyToken(token: string): any {
        return jwt.verify(token, JWT_SECRET);
    },
};
