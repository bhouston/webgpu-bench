import { expect, test, vi } from 'vitest';
import { createPipeline } from './common.ts';

test('pipeline creation propagates async compile errors with the benchmark label', async () => {
  const device = {
    pushErrorScope: vi.fn(),
    createShaderModule: vi.fn(() => ({})),
    popErrorScope: vi.fn(async () => null),
    createComputePipelineAsync: vi.fn(async () => {
      throw new Error('bad shader');
    }),
  };
  await expect(createPipeline(device as unknown as GPUDevice, 'example', 'bad WGSL')).rejects.toThrow(
    'Failed to create pipeline "example": bad shader',
  );
  expect(device.popErrorScope).toHaveBeenCalledOnce();
});

test('shader-module validation is not lost when an async pipeline resolves', async () => {
  const device = {
    pushErrorScope: vi.fn(),
    createShaderModule: vi.fn(() => ({})),
    popErrorScope: vi.fn(async () => ({ message: 'invalid module' })),
    createComputePipelineAsync: vi.fn(async () => ({})),
  };
  await expect(createPipeline(device as unknown as GPUDevice, 'example', 'bad WGSL')).rejects.toThrow('invalid module');
});
