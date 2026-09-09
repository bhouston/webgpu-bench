import { test, expect } from 'vitest';

test('navigator.gpu is available', async () => {
  expect(navigator.gpu).toBeDefined();
  const adapter = await navigator.gpu.requestAdapter();
  expect(adapter).not.toBeNull();
});
