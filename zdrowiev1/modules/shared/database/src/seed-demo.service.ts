import { Injectable, OnModuleInit } from '@nestjs/common';
import { UserService } from '@monorepo/user';
import { AuthService } from '@monorepo/auth';

@Injectable()
export class SeedDemoService implements OnModuleInit {
  constructor(
    private readonly userService: UserService,
    private readonly authService: AuthService,
  ) {}

  async onModuleInit() {
    if (process.env.NODE_ENV !== 'production') {
      await this.seed();
    }
  }

  async seed() {
    const demoEmail = 'demo@example.com';
    const exists = await this.userService.findByEmail(demoEmail);

    if (!exists) {
      console.log('----- CREATING DEMO DATA -----');
      await this.authService.register({
        email: demoEmail,
        password: 'Password123!',
        firstName: 'Demo',
        lastName: 'User',
        isDemo: true,
      });
      console.log('----- DEMO USER CREATED: demo@example.com / Password123! -----');
    }
  }
}
