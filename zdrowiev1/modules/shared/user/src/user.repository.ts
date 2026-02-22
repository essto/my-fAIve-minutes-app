import { Injectable } from '@nestjs/common';
import { db, users } from '@monorepo/database';
import { eq } from 'drizzle-orm';
import { User } from '@monorepo/shared-types';

@Injectable()
export class UserRepository {
  constructor() {}

  async findById(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user as User | undefined;
  }

  async findByEmail(email: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.email, email));
    return user as User | undefined;
  }

  async create(data: Partial<User>): Promise<User> {
    const [user] = await db
      .insert(users)
      .values(data as any)
      .returning();
    return user as User;
  }

  async update(id: string, data: Partial<User>): Promise<User> {
    const [user] = await db
      .update(users)
      .set(data as any)
      .where(eq(users.id, id))
      .returning();
    return user as User;
  }

  async delete(id: string): Promise<void> {
    await db.delete(users).where(eq(users.id, id));
  }
}
