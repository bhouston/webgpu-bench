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
 * GPU timestamps are cross-checked against wall clock on every measurement.
 * Wall clock minus the measured fixed submit/readback overhead is what the
 * GPU actually spent, so a GPU reading under half of that is a broken timer.
 * Safari's `timestamp-query` does exactly this — reporting ~1ms for a ~70ms
 * batch — and trusting it makes calibration batch ~70x too much work into
 * one command buffer, which hangs the browser. Two consecutive failures are
 * required before demoting for good, so a one-off main-thread stall that
 * inflates wall time (GC, a compositor frame) can't cost a kernel its
 * accurate timer.
 */
const TIMESTAMP_CHECK_MIN_GPU_MS = 4;
const TIMESTAMP_MAX_UNDERREPORT = 2;
const TIMESTAMP_STRIKES_TO_DEMOTE = 2;
/** Empty submits measured at calibration to estimate the fixed per-measurement overhead. */
const OVERHEAD_PROBES = 3;

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
  /** Cleared once the GPU timestamps have disagreed with wall clock `TIMESTAMP_STRIKES_TO_DEMOTE` times running. */
  private trustTimestamps: boolean;
  private timestampStrikes = 0;
  /** Wall time of an empty submit (min of `OVERHEAD_PROBES`): submit, scheduling and readback, no GPU work. */
  private overheadMs = 0;

  constructor(private readonly h: KernelHarness) {
    this.timer = new GpuTimer(h.device, h.useTimestamps);
    this.trustTimestamps = this.timer.supported;
  }

  get timingMethod(): TimingMethod {
    return this.trustTimestamps ? 'gpu-timestamp' : 'cpu-wallclock';
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
   *
   * Batch sizing uses *wall-clock* time, never the GPU timestamps: wall
   * clock includes submit/readback overhead so it can only over-estimate,
   * which errs toward smaller batches — whereas a broken timestamp (see
   * `TIMESTAMP_MAX_UNDERREPORT`) would size a batch tens of times too large
   * and hang the browser. The work ramp does use the GPU time while it's
   * trusted (a single dispatch's wall time is mostly readback overhead, so
   * sizing from it leaves dispatches far too small), which is safe there
   * because growth is capped at 4x per step: a bogus reading can at worst
   * overshoot one step before the wall-clock cross-check demotes it.
   */
  async calibrate(): Promise<void> {
    const targetMs = this.h.targetMs ?? DEFAULT_MEASUREMENT.targetMs;
    const targetDispatchMs = this.h.targetDispatchMs ?? DEFAULT_MEASUREMENT.targetDispatchMs;
    const warmups = this.h.warmups ?? DEFAULT_MEASUREMENT.warmups;
    const maxIterations = this.h.maxIterations ?? DEFAULT_MEASUREMENT.maxIterations;

    this.overheadMs = Number.POSITIVE_INFINITY;
    for (let i = 0; i < OVERHEAD_PROBES; i++) {
      this.overheadMs = Math.min(this.overheadMs, (await this.measureRaw(0)).wallMs);
    }

    const knob = this.h.work;
    if (knob) {
      let work = Math.max(1, Math.floor(knob.min));
      knob.apply(work);
      const dispatchTime = async () => {
        const { gpuMs, wallMs } = await this.measureRaw(1);
        return gpuMs ?? wallMs;
      };
      let dispatchMs = await dispatchTime();
      // Ramp: the dispatch time is ~linear in work, so extrapolate straight
      // to the target, but cap the growth per step and re-measure.
      while (work < knob.max && dispatchMs < targetDispatchMs) {
        const wanted = (work * targetDispatchMs) / Math.max(dispatchMs, 1e-6);
        const next = Math.min(knob.max, Math.floor(Math.min(wanted, work * MAX_PROBE_GROWTH)));
        if (next <= work) break;
        work = next;
        knob.apply(work);
        dispatchMs = await dispatchTime();
      }
      this.currentWork = work;
    }

    // Batch sizing: grow from a single dispatch until one measurement's wall
    // time is near the target. Wall clock over-estimates a tiny dispatch
    // (fixed submit/readback overhead dominates), so the first extrapolation
    // under-shoots and a couple of refinement steps home in from below. The
    // very first measurement never ends the search: it can absorb a one-off
    // stall (lazy shader compile, GC) that would leave the batch at 1.
    let iterations = 1;
    for (let step = 0; step < 4; step++) {
      const { wallMs } = await this.measureRaw(iterations);
      if ((step > 0 && wallMs >= targetMs / 2) || iterations >= maxIterations) break;
      const next = Math.min(maxIterations, Math.floor((iterations * targetMs) / Math.max(wallMs, 1e-6)));
      if (next <= iterations) break;
      iterations = next;
    }
    this.iterations = iterations;

    for (let i = 0; i < warmups; i++) {
      await this.measureRaw(this.iterations);
    }
  }

  /** One timed measurement: per-op time in ms. Requires `calibrate()` first. */
  async sample(): Promise<number> {
    if (this.iterations === 0) {
      throw new Error('KernelSampler.sample() called before calibrate()');
    }
    const { gpuMs, wallMs } = await this.measureRaw(this.iterations);
    return (gpuMs ?? wallMs) / this.iterations;
  }

  destroy(): void {
    this.timer.destroy();
  }

  /**
   * Submits `iterations` dispatches as one command buffer and waits for
   * them. Always returns the wall-clock time; returns the GPU-timestamp
   * time too while the timestamps are trusted. An implausibly small reading
   * (see `TIMESTAMP_MAX_UNDERREPORT`) is replaced by wall clock, and the
   * sampler is demoted to wall-clock for good on the second in a row.
   */
  private async measureRaw(iterations: number): Promise<{ gpuMs: number | null; wallMs: number }> {
    const { h, timer } = this;
    const useTimestamps = this.trustTimestamps;
    const encoder = h.device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: useTimestamps ? timer.timestampWrites : undefined });
    h.encode(pass, iterations);
    pass.end();
    if (useTimestamps) timer.resolve(encoder);

    const cpuStart = performance.now();
    h.device.queue.submit([encoder.finish()]);

    if (!useTimestamps) {
      await h.device.queue.onSubmittedWorkDone();
      return { gpuMs: null, wallMs: performance.now() - cpuStart };
    }

    const gpuMs = await timer.readElapsedMs();
    const wallMs = performance.now() - cpuStart;
    const gpuWallMs = wallMs - this.overheadMs;
    if (
      iterations > 0 &&
      gpuWallMs >= TIMESTAMP_CHECK_MIN_GPU_MS &&
      !(gpuMs * TIMESTAMP_MAX_UNDERREPORT >= gpuWallMs)
    ) {
      this.timestampStrikes += 1;
      if (this.timestampStrikes >= TIMESTAMP_STRIKES_TO_DEMOTE) this.trustTimestamps = false;
      return { gpuMs: null, wallMs };
    }
    this.timestampStrikes = 0;
    return { gpuMs, wallMs };
  }
}
