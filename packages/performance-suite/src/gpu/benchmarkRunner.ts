import { GpuTimer } from './timing.ts';
import type { TimingMethod } from '../types.ts';

/** Knobs controlling how one timed measurement of a kernel is taken. */
export interface MeasurementConfig {
  /**
   * Target duration of a single measurement (one submitted command buffer),
   * in ms; sets how many dispatches get batched. Default 100.
   */
  targetMs?: number;
  /**
   * Target duration of a single *dispatch*, in ms. A dispatch can't be
   * preempted, so a long one freezes the display; kernels that expose a
   * work knob get it calibrated so one dispatch lands near this. Default 10.
   */
  targetDispatchMs?: number;
  /** Discarded measurements taken after calibration and before timing starts. Default 1. */
  warmups?: number;
  /** Hard cap on how many dispatches get batched into one measurement (guards against runaway calibration). */
  maxIterations?: number;
}

export const DEFAULT_MEASUREMENT: Required<MeasurementConfig> = {
  targetMs: 100,
  targetDispatchMs: 10,
  warmups: 1,
  maxIterations: 200_000,
};

/**
 * A kernel's tunable per-dispatch work (for the raw-FLOPS kernels: the
 * in-shader loop trip count). Lets the sampler size a single dispatch to
 * `targetDispatchMs` on whatever GPU it finds itself on, instead of a fixed
 * count that's a few ms on a desktop and a second on a phone.
 */
export interface WorkKnob {
  /** Smallest sensible work per dispatch (also the first calibration probe, so keep it cheap). */
  min: number;
  /** Largest work per dispatch (the kernel's own default; bounds numerical range and per-thread setup amortisation). */
  max: number;
  /** Make subsequent dispatches use this much work (e.g. rewrite the params uniform). */
  apply(work: number): void;
}

export interface KernelHarness extends MeasurementConfig {
  device: GPUDevice;
  /** Records `iterations` back-to-back dispatches (same pipeline/bind group) into the pass. */
  encode: (pass: GPUComputePassEncoder, iterations: number) => void;
  useTimestamps: boolean;
  /** Present for kernels whose per-dispatch work can be resized at runtime. */
  work?: WorkKnob;
}

/** Never grow a calibration probe by more than this factor per step, so no probe can run away on a slow GPU. */
const MAX_PROBE_GROWTH = 4;

/**
 * Takes individual timed measurements of one kernel on demand, so a
 * scheduler can interleave many kernels round-robin (see `runSampling`)
 * instead of hammering one kernel until it converges. Holds the GPU timer
 * and the calibrated batch size between calls.
 *
 * Lifecycle: `calibrate()` once (sizes the dispatch and the batch, runs the
 * warmups), then `sample()` as often as wanted, then `destroy()`.
 */
export class KernelSampler {
  private readonly timer: GpuTimer;
  private iterations = 0;
  private currentWork: number | undefined;

  constructor(private readonly h: KernelHarness) {
    this.timer = new GpuTimer(h.device, h.useTimestamps);
  }

  get timingMethod(): TimingMethod {
    return this.timer.supported ? 'gpu-timestamp' : 'cpu-wallclock';
  }

  /** Number of dispatches batched into each measurement; 0 until calibrated. */
  get innerIterations(): number {
    return this.iterations;
  }

  /** Per-dispatch work the kernel was calibrated to; undefined if it has no work knob. */
  get work(): number | undefined {
    return this.currentWork;
  }

  /**
   * Sizes the kernel so one dispatch takes about `targetDispatchMs` (when it
   * has a work knob: start from the cheapest probe and ramp up, never more
   * than 4x per step, so even a very slow GPU never gets handed a long
   * dispatch), then batches dispatches to fill `targetMs`, then runs the
   * discarded warmups. The probes double as pipeline warm-up.
   */
  async calibrate(): Promise<void> {
    const targetMs = this.h.targetMs ?? DEFAULT_MEASUREMENT.targetMs;
    const targetDispatchMs = this.h.targetDispatchMs ?? DEFAULT_MEASUREMENT.targetDispatchMs;
    const warmups = this.h.warmups ?? DEFAULT_MEASUREMENT.warmups;
    const maxIterations = this.h.maxIterations ?? DEFAULT_MEASUREMENT.maxIterations;

    let dispatchMs: number;
    const knob = this.h.work;
    if (knob) {
      let work = Math.max(1, Math.floor(knob.min));
      knob.apply(work);
      dispatchMs = await this.measureOnce(1);
      // Ramp: the dispatch time is ~linear in work, so extrapolate straight
      // to the target, but cap the growth per step and re-measure.
      while (work < knob.max && dispatchMs < targetDispatchMs) {
        const wanted = (work * targetDispatchMs) / Math.max(dispatchMs, 1e-6);
        const next = Math.min(knob.max, Math.floor(Math.min(wanted, work * MAX_PROBE_GROWTH)));
        if (next <= work) break;
        work = next;
        knob.apply(work);
        dispatchMs = await this.measureOnce(1);
      }
      this.currentWork = work;
    } else {
      dispatchMs = await this.measureOnce(1);
    }

    const rawIterations = Math.round(targetMs / Math.max(dispatchMs, 1e-6));
    this.iterations = Math.min(maxIterations, Math.max(1, rawIterations));

    for (let i = 0; i < warmups; i++) {
      await this.measureOnce(this.iterations);
    }
  }

  /** One timed measurement: per-op time in ms. Requires `calibrate()` first. */
  async sample(): Promise<number> {
    if (this.iterations === 0) {
      throw new Error('KernelSampler.sample() called before calibrate()');
    }
    return this.measureOnce(this.iterations);
  }

  destroy(): void {
    this.timer.destroy();
  }

  /** Submits `iterations` dispatches as one command buffer and returns the per-dispatch time in ms. */
  private async measureOnce(iterations: number): Promise<number> {
    const { h, timer } = this;
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
}
