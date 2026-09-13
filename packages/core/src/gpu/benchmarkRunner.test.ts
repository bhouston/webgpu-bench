import { expect, test, vi } from 'vitest';
import { KernelSampler } from './benchmarkRunner.ts';

function fakeSampler(wallTime: (iterations: number) => number, maxIterations = 200_000) {
  const sampler = new KernelSampler({
    device: {} as GPUDevice,
    encode() {},
    useTimestamps: false,
    targetMs: 20,
    warmups: 0,
    maxIterations,
  });
  // Keep the real calibration algorithm; replace only the GPU measurement.
  const measure = vi
    .spyOn(
      sampler as unknown as { measureRaw(iterations: number): Promise<{ gpuMs: null; wallMs: number }> },
      'measureRaw',
    )
    .mockImplementation(async (iterations) => ({ gpuMs: null, wallMs: wallTime(iterations) }));
  return { sampler, measure };
}

test('sub-millisecond probes cannot jump to a huge batch on a coarse clock', async () => {
  const { sampler, measure } = fakeSampler((iterations) => Math.floor(iterations * 0.01));
  await sampler.calibrate();
  const batches = measure.mock.calls.map(([iterations]) => iterations).filter(Boolean);
  expect(batches[0]).toBe(1);
  for (let i = 1; i < batches.length; i++) expect(batches[i]!).toBeLessThanOrEqual(batches[i - 1]! * 4);
  expect(sampler.innerIterations).toBeLessThan(4096);
  // Calibration must still grow enough to obtain a measurable sample.
  expect(sampler.innerIterations * 0.01).toBeGreaterThanOrEqual(10);
  expect(await sampler.sample()).toBeGreaterThan(0);
});

test('tiny nonzero readings also have bounded growth and respect the batch cap', async () => {
  const { sampler, measure } = fakeSampler(() => 0.001, 10);
  await sampler.calibrate();
  expect(measure.mock.calls.map(([iterations]) => iterations).filter(Boolean)).toEqual([1, 4, 10]);
  expect(sampler.innerIterations).toBe(10);
});

test('a single over-budget operation is never multiplied into a larger probe', async () => {
  const { sampler, measure } = fakeSampler((iterations) => iterations * 500);
  await sampler.calibrate();
  expect(measure.mock.calls.map(([iterations]) => iterations).filter(Boolean)).toEqual([1]);
  expect(sampler.innerIterations).toBe(1);
});
