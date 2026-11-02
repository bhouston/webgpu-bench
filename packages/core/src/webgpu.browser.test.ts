import { test, expect } from 'vitest';

test('navigator.gpu is available', async () => {
  expect(navigator.gpu).toBeDefined();
  const adapter = await navigator.gpu.requestAdapter();
  expect(adapter).not.toBeNull();
  console.log(
    'WebGPU adapter:',
    adapter!.info.vendor,
    adapter!.info.architecture,
    adapter!.info.description,
    adapter!.info.isFallbackAdapter,
  );
});
