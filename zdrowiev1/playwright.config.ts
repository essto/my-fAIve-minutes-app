import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './apps',
  testMatch: '**/*.spec.ts',
  use: {
    baseURL: 'http://localhost:3005',
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
});
