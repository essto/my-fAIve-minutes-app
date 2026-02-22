import { Injectable, Inject } from '@nestjs/common';
import { UserRepository } from './user.repository';
import { User } from '@monorepo/shared-types';

@Injectable()
export class UserService {
  constructor(
    @Inject(UserRepository)
    private readonly userRepository: UserRepository,
  ) {}

  async findById(id: string): Promise<User | undefined> {
    return this.userRepository.findById(id);
  }

  async findByEmail(email: string): Promise<User | undefined> {
    return this.userRepository.findByEmail(email);
  }

  async create(data: Partial<User>): Promise<User> {
    return this.userRepository.create(data);
  }

  async update(id: string, data: Partial<User>): Promise<User> {
    return this.userRepository.update(id, data);
  }
}
