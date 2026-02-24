import { vi } from 'vitest';
import { Test, TestingModule } from '@nestjs/testing';
import { SeedDemoService } from '../seed-demo.service';
import { UserService } from '@monorepo/user';
import { AuthService } from '@monorepo/auth';

describe('SeedDemoService', () => {
    let service: SeedDemoService;
    let userService: UserService;
    let authService: AuthService;

    const mockUserService = {
        findByEmail: vi.fn(),
    };

    const mockAuthService = {
        register: vi.fn(),
    };

    beforeEach(async () => {
        // Reset env var
        process.env.NODE_ENV = 'development';

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                SeedDemoService,
                {
                    provide: UserService,
                    useValue: mockUserService,
                },
                {
                    provide: AuthService,
                    useValue: mockAuthService,
                },
            ],
        }).compile();

        service = module.get<SeedDemoService>(SeedDemoService);
        userService = module.get<UserService>(UserService);
        authService = module.get<AuthService>(AuthService);
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    describe('onModuleInit', () => {
        it('should call seed() if NODE_ENV is not production', async () => {
            process.env.NODE_ENV = 'development';
            const seedSpy = vi.spyOn(service, 'seed').mockResolvedValue(undefined);

            await service.onModuleInit();

            expect(seedSpy).toHaveBeenCalled();
        });

        it('should NOT call seed() if NODE_ENV is production', async () => {
            process.env.NODE_ENV = 'production';
            const seedSpy = vi.spyOn(service, 'seed').mockResolvedValue(undefined);

            await service.onModuleInit();

            expect(seedSpy).not.toHaveBeenCalled();
        });
    });

    describe('seed', () => {
        it('should not register demo user if already exists', async () => {
            mockUserService.findByEmail.mockResolvedValue({ id: 'exists' });

            await service.seed();

            expect(userService.findByEmail).toHaveBeenCalledWith('demo@example.com');
            expect(authService.register).not.toHaveBeenCalled();
        });

        it('should register demo user if does not exist', async () => {
            mockUserService.findByEmail.mockResolvedValue(null);

            await service.seed();

            expect(authService.register).toHaveBeenCalledWith({
                email: 'demo@example.com',
                password: 'Password123!',
                firstName: 'Demo',
                lastName: 'User',
                isDemo: true,
            });
        });
    });
});
