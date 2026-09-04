import { GpuTimer } from './timing.ts';
import type { SamplingStopReason, Stats, TimingMethod } from '../types.ts';
import { computeStats, hasConverged } from '../stats.ts';

/** Knobs controlling how many timed measurements a benchmark takes. */
export interface SamplingConfig {
  /** Target duration of a single measurement, in ms; sets how many dispatches get batched. Default 300. */
  targetMs?: number;
  /** Discarded measurements taken after calibration and before timing starts. Default 1. */
  warmups?: number;
  /** Fewest timed measurements before convergence can be declared. Default 3. */
  minRuns?: number;
  /** Most timed measurements taken even if the timings never converge. Default 10. */
  maxRuns?: number;
  /** Convergence target: 95% CI half-width on the mean as a fraction of the mean. Default 0.03. */
  precision?: number;
}

export const DEFAULT_SAMPLING: Required<SamplingConfig> = {
  targetMs: 300,
  warmups: 1,
  minRuns: 3,
  maxRuns: 10,
  precision: 0.03,
};

export interface KernelHarness extends SamplingConfig {
  device: GPUDevice;
  /** Records `iterations` back-to-back dispatches (same pipeline/bind group) into the pass. */
  encode: (pass: GPUComputePassEncoder, iterations: number) => void;
  useTimestamps: boolean;
  /** Hard cap on how many dispatches get batched into one measurement (guards against runaway calibration). */
  maxIterations?: number;
}

export interface TimedRun {
  timingMethod: TimingMethod;
  innerIterations: number;
  /** Per-op time in ms, one entry per timed run. */
  timesMs: number[];
  stats: Stats;
  stopReason: SamplingStopReason;
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
 * measurements, then samples adaptively: at least `minRuns` timed
 * measurements, stopping as soon as the 95% confidence interval on the mean
 * is within `precision` of the mean (see `hasConverged`), and never more than
 * `maxRuns` so a noisy device can't stall the suite.
 */
export async function runTimedBenchmark(h: KernelHarness): Promise<TimedRun> {
  const targetMs = h.targetMs ?? DEFAULT_SAMPLING.targetMs;
  const warmups = h.warmups ?? DEFAULT_SAMPLING.warmups;
  const minRuns = Math.max(1, h.minRuns ?? DEFAULT_SAMPLING.minRuns);
  const maxRuns = Math.max(minRuns, h.maxRuns ?? DEFAULT_SAMPLING.maxRuns);
  const precision = h.precision ?? DEFAULT_SAMPLING.precision;
  const maxIterations = h.maxIterations ?? 200_000;

  const timer = new GpuTimer(h.device, h.useTimestamps);
  try {
    // Calibration: time a small batch, then scale up to hit the target duration.
    // (This also serves as a first, untimed warm-up of the pipeline.)
    const calibrationIterations = 4;
    const perOpMsEstimate = await measureOnce(h, calibrationIterations, timer);
    const rawIterations = Math.round(targetMs / Math.max(perOpMsEstimate, 1e-6));
    const iterations = Math.min(maxIterations, Math.max(1, rawIterations));

    for (let i = 0; i < warmups; i++) {
      await measureOnce(h, iterations, timer);
    }

    const timesMs: number[] = [];
    let stopReason: SamplingStopReason = 'max-runs';
    while (timesMs.length < maxRuns) {
      timesMs.push(await measureOnce(h, iterations, timer));
      if (hasConverged(timesMs, { minRuns, precision })) {
        stopReason = 'converged';
        break;
      }
    }

    return {
      timingMethod: timer.supported ? 'gpu-timestamp' : 'cpu-wallclock',
      innerIterations: iterations,
      timesMs,
      stats: computeStats(timesMs),
      stopReason,
    };
  } finally {
    timer.destroy();
  }
}
