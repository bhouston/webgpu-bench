import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    projects: [
      'packages/core/vitest.config.ts',
      { test: { name: 'cli', root: 'packages/cli', include: ['src/**/*.test.ts'] } },
    ],
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      exclude: ['**/*.test.ts'],
      reporter: ['text', 'json-summary', 'lcov', 'html'],
      thresholds: { statements: 60, branches: 60, functions: 60, lines: 60 },
    },
  },
});
