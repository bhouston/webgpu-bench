import { test, expect } from 'vitest';
import { prepareFlopsF32Rsqrt } from './benchmarks/flopsMath.ts';
import { KernelSampler } from './gpu/benchmarkRunner.ts';
import { acquireGpuContext } from './gpu/context.ts';

// Template for a fast, single-kernel compile+run check to write alongside a
// new shader — fails in under a second instead of after the whole suite's
// round-robin scheduler runs. See docs/vitest-browser-migration.md.
test('f32-rsqrt compiles and runs', async () => {
  const ctx = await acquireGpuContext();
  try {
    const prepared = await prepareFlopsF32Rsqrt(ctx, { iterations: 16, threads: 1024 });
    expect(prepared.kind).not.toBe('skipped');
    if (prepared.kind !== 'kernel') return;
    const sampler = new KernelSampler(prepared.harness);
    await sampler.calibrate();
    const ms = await sampler.sample();
    expect(Number.isFinite(ms)).toBe(true);
    expect(ms).toBeGreaterThan(0);
  } finally {
    ctx.device.destroy();
  }
});
