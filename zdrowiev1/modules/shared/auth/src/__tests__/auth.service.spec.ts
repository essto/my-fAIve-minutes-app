import { Test, TestingModule } from '@nestjs/testing';
import { JwtService } from '@nestjs/jwt';
import { AuthService } from '../auth.service';
import { UserService } from '@monorepo/user';
import { vi, describe, it, expect, beforeEach } from 'vitest';

describe('AuthService', () => {
  let service: AuthService;
  let userService: UserService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AuthService,
        {
          provide: UserService,
          useValue: {
            findByEmail: vi.fn(),
            create: vi.fn(),
          },
        },
        {
          provide: JwtService,
          useValue: {
            sign: vi.fn().mockReturnValue('mock-token'),
          },
        },
      ],
    }).compile();

    service = module.get<AuthService>(AuthService);
    userService = module.get<UserService>(UserService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('hashPassword', () => {
    it('should hash password correctly', async () => {
      const password = 'Password123!';
      const hash = await service.hashPassword(password);
      expect(hash).not.toEqual(password);
      expect(hash.length).toBeGreaterThan(20);
    });
  });

  describe('validateUser', () => {
    it('should return user without password if valid', async () => {
      const password = 'Password123!';
      const hashedPassword = await service.hashPassword(password);
      const mockUser = { id: '1', email: 'test@example.com', password: hashedPassword };

      vi.spyOn(userService, 'findByEmail').mockResolvedValue(mockUser as any);

      const result = await service.validateUser('test@example.com', password);
      expect(result).toBeDefined();
      expect(result?.email).toBe('test@example.com');
      expect((result as any).password).toBeUndefined();
    });

    it('should return null if invalid password', async () => {
      const mockUser = { id: '1', email: 'test@example.com', password: 'hashed' };
      vi.spyOn(userService, 'findByEmail').mockResolvedValue(mockUser as any);

      const result = await service.validateUser('test@example.com', 'wrong');
      expect(result).toBeNull();
    });
  });
});
