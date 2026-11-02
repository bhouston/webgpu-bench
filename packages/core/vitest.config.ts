import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

const ci = !!process.env.CI;
// Distro-dependent name: lvp_icd.json (newer Ubuntu) or lvp_icd.x86_64.json (older).
const lavapipeIcd = ['lvp_icd.json', 'lvp_icd.x86_64.json']
  .map((f) => `/usr/share/vulkan/icd.d/${f}`)
  .find((f) => existsSync(f));
// CI: headed Chromium under xvfb on Mesa lavapipe (the three.js method), not SwiftShader.
const chromiumArgs = ci
  ? [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--disable-vulkan-surface',
      '--ignore-gpu-blocklist',
      '--disable-gpu-driver-bug-workarounds',
      '--disable-gpu-watchdog',
      '--no-sandbox',
    ]
  : ['--headless=new', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist'];

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  test: {
    fsModuleCache: true,
    name: 'core',
    projects: [
      {
        test: {
          fsModuleCache: true,
          name: 'node',
          include: ['src/**/*.test.ts'],
          exclude: ['src/**/*.browser.test.ts'],
        },
      },
      ...(['chromium', 'webkit'] as const).map((browser, index) => ({
        test: {
          fsModuleCache: true,
          name: `browser-${browser}`,
          include: ['src/**/*.browser.test.ts'],
          // GPU workloads must not compete with other test files or browsers.
          fileParallelism: false,
          sequence: { groupOrder: index + 1 },
          browser: {
            enabled: true,
            // Use the full Chromium binary's new headless mode for WebGPU
            // locally; CI runs it headed under xvfb.
            // WebKit uses its own headless mode, without Chromium flags.
            headless: browser === 'webkit',
            provider: playwright({
              launchOptions:
                browser === 'chromium'
                  ? {
                      args: chromiumArgs,
                      ...(ci && {
                        env: {
                          ...process.env,
                          VK_DRIVER_FILES: lavapipeIcd ?? '',
                        },
                      }),
                    }
                  : {},
            }),
            instances: [{ browser }],
          },
        },
      })),
    ],
  },
});
