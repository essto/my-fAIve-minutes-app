import { vi } from 'vitest';
import { Test, TestingModule } from '@nestjs/testing';
import { UserService } from '../user.service';
import { UserRepository } from '../user.repository';
import { User } from '@monorepo/shared-types';

describe('UserService', () => {
  let service: UserService;
  let repository: UserRepository;

  const mockUser: User = {
    id: '1',
    email: 'test@example.com',
    password: 'hashedpassword123',
    firstName: 'Testy',
    lastName: 'McTestface',
    isDemo: false,
    createdAt: new Date(),
    updatedAt: new Date(),
  };

  const mockUserRepository = {
    findById: vi.fn(),
    findByEmail: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
    delete: vi.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        UserService,
        {
          provide: UserRepository,
          useValue: mockUserRepository,
        },
      ],
    }).compile();

    service = module.get<UserService>(UserService);
    repository = module.get<UserRepository>(UserRepository);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('findById', () => {
    it('should return a user if found', async () => {
      mockUserRepository.findById.mockResolvedValue(mockUser);
      const result = await service.findById('1');
      expect(result).toEqual(mockUser);
      expect(repository.findById).toHaveBeenCalledWith('1');
    });

    it('should return undefined if user not found', async () => {
      mockUserRepository.findById.mockResolvedValue(undefined);
      const result = await service.findById('999');
      expect(result).toBeUndefined();
    });
  });

  describe('findByEmail', () => {
    it('should return a user by email', async () => {
      mockUserRepository.findByEmail.mockResolvedValue(mockUser);
      const result = await service.findByEmail('test@example.com');
      expect(result).toEqual(mockUser);
      expect(repository.findByEmail).toHaveBeenCalledWith('test@example.com');
    });
  });

  describe('create', () => {
    it('should create a new user', async () => {
      const newUser = { email: 'new@example.com', firstName: 'New' };
      mockUserRepository.create.mockResolvedValue({ ...mockUser, ...newUser, id: '2' });

      const result = await service.create(newUser);
      expect(result.email).toBe('new@example.com');
      expect(repository.create).toHaveBeenCalledWith(newUser);
    });
  });

  describe('update', () => {
    it('should update an existing user', async () => {
      const updateData = { firstName: 'UpdatedName' };
      mockUserRepository.update.mockResolvedValue({ ...mockUser, ...updateData });

      const result = await service.update('1', updateData);
      expect(result.firstName).toBe('UpdatedName');
      expect(repository.update).toHaveBeenCalledWith('1', updateData);
    });
  });
});
