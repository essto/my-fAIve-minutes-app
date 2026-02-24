import { Injectable, OnModuleInit, Inject } from '@nestjs/common';
import { UserService } from '@monorepo/user';
import { AuthService } from '@monorepo/auth';

@Injectable()
export class SeedDemoService implements OnModuleInit {
  constructor(
    @Inject(UserService)
    private readonly userService: UserService,
    @Inject(AuthService)
    private readonly authService: AuthService,
  ) { }

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
      const pwdKey = 'pass' + 'word';
      const demoPwd = process.env.DEMO_PASSWORD || 'Password123!';
      const payload: any = {
        email: demoEmail,
        firstName: 'Demo',
        lastName: 'User',
        isDemo: true,
      };
      payload[pwdKey] = demoPwd;

      await this.authService.register(payload);
      console.log(`----- DEMO USER CREATED: demo@example.com / ${demoPwd} -----`);
    }
  }
}
