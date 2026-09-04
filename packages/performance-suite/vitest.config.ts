import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'node',
          include: ['src/**/*.test.ts'],
          exclude: ['src/**/*.browser.test.ts'],
        },
      },
      {
        test: {
          name: 'browser',
          include: ['src/**/*.browser.test.ts'],
          browser: {
            enabled: true,
            provider: playwright(),
            headless: false,
            instances: [
              {
                browser: 'chromium',
                launch: {
                  // Playwright's `headless: true` launches the stripped
                  // "headless shell" binary, which has no GPU process and so
                  // no WebGPU. `--headless=new` on the full Chrome binary
                  // (headless: false here) is the real new-headless mode and
                  // does support WebGPU.
                  args: ['--headless=new', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
                },
              },
            ],
          },
        },
      },
    ],
  },
});
