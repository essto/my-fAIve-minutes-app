import { JwtStrategy } from '../jwt.strategy';

describe('JwtStrategy', () => {
    let strategy: JwtStrategy;

    beforeEach(() => {
        // Override env var for test determinism
        process.env.JWT_SECRET = 'test-secret';
        strategy = new JwtStrategy();
    });

    afterEach(() => {
        delete process.env.JWT_SECRET;
    });

    it('should be defined', () => {
        expect(strategy).toBeDefined();
    });

    describe('validate', () => {
        it('should extract userId and email from payload', async () => {
            const payload = { sub: 'user123', email: 'test@example.com', iat: 12345 };
            const result = await strategy.validate(payload);

            expect(result).toEqual({ userId: 'user123', email: 'test@example.com' });
        });
    });
});
