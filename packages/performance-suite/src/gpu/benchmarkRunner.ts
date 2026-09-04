import { GpuTimer } from './timing.ts';
import type { Stats, TimingMethod } from '../types.ts';
import { computeStats } from '../stats.ts';

export interface KernelHarness {
  device: GPUDevice;
  /** Records `iterations` back-to-back dispatches (same pipeline/bind group) into the pass. */
  encode: (pass: GPUComputePassEncoder, iterations: number) => void;
  useTimestamps: boolean;
  targetMs?: number;
  warmups?: number;
  runs?: number;
  /** Hard cap on how many dispatches get batched into one measurement (guards against runaway calibration). */
  maxIterations?: number;
}

export interface TimedRun {
  timingMethod: TimingMethod;
  innerIterations: number;
  /** Per-op time in ms, one entry per timed run. */
  timesMs: number[];
  stats: Stats;
}

async function measureOnce(h: KernelHarness, iterations: number, timer: GpuTimer): Promise<number> {
  const encoder = h.device.createCommandEncoder();
  const pass = encoder.beginComputePass({ timestampWrites: timer.timestampWrites });
  h.encode(pass, iterations);
  pass.end();
  timer.resolve(encoder);

  const cpuStart = performance.now();
  h.device.queue.submit([encoder.finish()]);

  if (timer.supported) {
    const totalMs = await timer.readElapsedMs();
    return totalMs / iterations;
  }
  await h.device.queue.onSubmittedWorkDone();
  const totalMs = performance.now() - cpuStart;
  return totalMs / iterations;
}

/**
 * Calibrates how many dispatches to batch into one measurement (so a single
 * measurement takes roughly `targetMs`), runs `warmups` throwaway
 * measurements, then `runs` timed measurements, returning per-op times.
 */
export async function runTimedBenchmark(h: KernelHarness): Promise<TimedRun> {
  const targetMs = h.targetMs ?? 300;
  const warmups = h.warmups ?? 3;
  const runs = h.runs ?? 10;
  const maxIterations = h.maxIterations ?? 200_000;

  const timer = new GpuTimer(h.device, h.useTimestamps);
  try {
    // Calibration: time a small batch, then scale up to hit the target duration.
    const calibrationIterations = 4;
    const perOpMsEstimate = await measureOnce(h, calibrationIterations, timer);
    const rawIterations = Math.round(targetMs / Math.max(perOpMsEstimate, 1e-6));
    const iterations = Math.min(maxIterations, Math.max(1, rawIterations));

    for (let i = 0; i < warmups; i++) {
      await measureOnce(h, iterations, timer);
    }

    const timesMs: number[] = [];
    for (let i = 0; i < runs; i++) {
      timesMs.push(await measureOnce(h, iterations, timer));
    }

    return {
      timingMethod: timer.supported ? 'gpu-timestamp' : 'cpu-wallclock',
      innerIterations: iterations,
      timesMs,
      stats: computeStats(timesMs),
    };
  } finally {
    timer.destroy();
  }
}
