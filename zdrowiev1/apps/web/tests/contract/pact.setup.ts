import { PactV3 } from '@pact-foundation/pact';
import path from 'path';

export const provider = new PactV3({
    consumer: 'web',
    provider: 'api',
    dir: path.resolve(process.cwd(), '../../pacts'),
    logLevel: 'info'
});
