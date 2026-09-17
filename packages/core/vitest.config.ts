import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  test: {
    name: 'core',
    projects: [
      {
        test: {
          name: 'node',
          include: ['src/**/*.test.ts'],
          exclude: ['src/**/*.browser.test.ts'],
        },
      },
      ...(['chromium', 'webkit'] as const).map((browser, index) => ({
        test: {
          name: `browser-${browser}`,
          include: ['src/**/*.browser.test.ts'],
          // GPU workloads must not compete with other test files or browsers.
          fileParallelism: false,
          sequence: { groupOrder: index + 1 },
          browser: {
            enabled: true,
            // Use the full Chromium binary's new headless mode for WebGPU.
            // WebKit uses its own headless mode, without Chromium flags.
            headless: browser === 'webkit',
            provider: playwright({
              launchOptions:
                browser === 'chromium'
                  ? { args: ['--headless=new', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] }
                  : {},
            }),
            instances: [{ browser }],
          },
        },
      })),
    ],
  },
});
