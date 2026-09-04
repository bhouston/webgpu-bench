import { computeStats, isBestStable, isThrottled } from './stats.ts';
import type { SamplingStopReason, Stats, SuiteProgressEvent } from './types.ts';

/** Knobs for the round-robin, best-of-N sampling scheduler. */
export interface SamplingConfig {
  /** Fewest kept measurements per benchmark before it can converge. Default 3. */
  minRounds?: number;
  /** Most kept measurements per benchmark. Default 10. */
  maxRounds?: number;
  /** Converged once the best run hasn't improved (by more than `improvementTolerance`) over this many kept measurements. Default 2. */
  stableRounds?: number;
  /** Relative improvement that counts as "still improving". Default 0.01. */
  improvementTolerance?: number;
  /** A measurement this fraction slower than the best is throttled and discarded. Default 0.20. */
  throttleThreshold?: number;
  /** Idle gap between consecutive measurements, in ms. Default 100. */
  idleMs?: number;
  /** Suite-wide pause when throttling is detected, in ms. Default 3000. */
  cooldownMs?: number;
  /** Cooldown pauses before giving up on still-unconverged benchmarks. Default 3. */
  maxCooldowns?: number;
  /** Fraction of a round's measurements (and at least two) that must be throttled to trigger a cooldown. Default 0.5. */
  throttledFraction?: number;
}

export const DEFAULT_SAMPLING: Required<SamplingConfig> = {
  minRounds: 3,
  maxRounds: 10,
  stableRounds: 2,
  improvementTolerance: 0.01,
  throttleThreshold: 0.2,
  idleMs: 100,
  cooldownMs: 3000,
  maxCooldowns: 3,
  throttledFraction: 0.5,
};

/** What the scheduler needs from a benchmark: something it can measure repeatedly. */
export interface Sampleable {
  id: string;
  /** One-time setup (calibration, warmups). Called before the first `sample()`. */
  calibrate(): Promise<void>;
  /** One timed measurement: per-op time in ms. */
  sample(): Promise<number>;
}

/** Running state of one benchmark inside the scheduler; surfaced after every measurement via `onUpdate`. */
export interface SampleState {
  id: string;
  /** Kept measurements, in order taken. */
  timesMs: number[];
  /** Measurements discarded as throttled, in order taken. */
  throttledMs: number[];
  /** Best kept measurement so far; Infinity before the first. */
  bestMs: number;
  stopReason?: SamplingStopReason;
  /** Set if calibration or a measurement threw; the benchmark is retired. */
  error?: unknown;
}

export interface SampleStateSnapshot extends SampleState {
  stats?: Stats;
}

export interface RunSamplingOptions extends SamplingConfig {
  /** Called after every measurement (and when a benchmark is retired) with that benchmark's state. */
  onUpdate?: (state: SampleStateSnapshot) => void | Promise<void>;
  onProgress?: (event: SuiteProgressEvent) => void;
  /** Injectable for tests; defaults to a real `setTimeout` sleep. */
  sleep?: (ms: number) => Promise<void>;
}

const isActive = (s: SampleState) => s.stopReason === undefined && s.error === undefined;

const realSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * Classifies one measurement against the benchmark's best-so-far and folds it
 * into the state. Pure apart from mutating `state`.
 *
 * Returns whether the sample was kept. A throttled sample (more than
 * `throttleThreshold` slower than the best) is recorded in `throttledMs` and
 * doesn't count toward convergence — it says something about the device's
 * temperature, not the kernel.
 */
export function recordSample(state: SampleState, ms: number, cfg: Required<SamplingConfig>): boolean {
  if (isThrottled(ms, state.bestMs, cfg.throttleThreshold)) {
    state.throttledMs.push(ms);
    // A benchmark that keeps coming back throttled while the rest of the
    // suite is fine (so no cooldown fires) must still terminate.
    if (state.timesMs.length + state.throttledMs.length >= 2 * cfg.maxRounds) state.stopReason = 'max-rounds';
    return false;
  }
  state.timesMs.push(ms);
  if (ms < state.bestMs) state.bestMs = ms;
  if (
    isBestStable(state.timesMs, {
      minRuns: cfg.minRounds,
      stableRuns: cfg.stableRounds,
      tolerance: cfg.improvementTolerance,
    })
  ) {
    state.stopReason = 'converged';
  } else if (state.timesMs.length >= cfg.maxRounds) {
    state.stopReason = 'max-rounds';
  }
  return true;
}

/**
 * Whether the measurements taken in one round add up to "the device is
 * throttled". Thermal throttling slows every kernel, so the signal is
 * agreement across benchmarks: at least two of them, and at least
 * `throttledFraction` of the round. One benchmark repeatedly coming in slow
 * while the rest are fine is that benchmark's own noise (Safari's
 * wall-clock timing of the write-bandwidth kernel jitters 15–40%), not
 * heat — pausing the suite for it would just waste time.
 */
export function roundIsThrottled(roundThrottled: readonly boolean[], cfg: Required<SamplingConfig>): boolean {
  const throttledCount = roundThrottled.filter(Boolean).length;
  return throttledCount >= 2 && throttledCount / roundThrottled.length >= cfg.throttledFraction;
}

function snapshot(state: SampleState): SampleStateSnapshot {
  return {
    ...state,
    timesMs: [...state.timesMs],
    throttledMs: [...state.throttledMs],
    stats: state.timesMs.length > 0 ? computeStats(state.timesMs) : undefined,
  };
}

/**
 * Round-robin, best-of-N sampling across many benchmarks.
 *
 * Rather than running each benchmark to convergence before starting the
 * next (which heats a phone up in seconds and leaves every later benchmark
 * measuring a throttled GPU), every round takes one short measurement of
 * every still-active benchmark, with an idle gap between measurements so
 * the GPU duty-cycles. Benchmarks retire as soon as their best run has
 * stopped improving (`isBestStable`). Measurements that come in well slower
 * than a benchmark's best are discarded as throttled; when a round shows
 * the device throttled overall — half the benchmarks slow, or any one slow
 * twice running — the whole suite pauses for `cooldownMs` and tries again,
 * up to `maxCooldowns` times, after which whatever is still unconverged is
 * reported with `stopReason: 'throttled'` alongside its best run so far.
 *
 * Only the best run is the headline number: every noise source makes runs
 * slower, never faster, so the minimum is the most robust estimate of what
 * the device can do.
 */
export async function runSampling(
  benchmarks: readonly Sampleable[],
  options: RunSamplingOptions = {},
): Promise<Map<string, SampleStateSnapshot>> {
  const cfg = resolveSamplingConfig(options);
  const sleep = options.sleep ?? realSleep;
  const onUpdate = options.onUpdate ?? (() => {});
  const onProgress = options.onProgress ?? (() => {});

  const states = new Map<string, SampleState>();
  const samplers = new Map<string, Sampleable>();
  for (const b of benchmarks) {
    states.set(b.id, { id: b.id, timesMs: [], throttledMs: [], bestMs: Number.POSITIVE_INFINITY });
    samplers.set(b.id, b);
  }
  let cooldowns = 0;
  let calibrated = false;
  for (let round = 1; ; round++) {
    const active = [...states.values()].filter(isActive);
    if (active.length === 0) break;
    onProgress({ type: 'round', round, active: active.length });

    const roundStates: SampleState[] = [];
    const roundThrottled: boolean[] = [];
    let first = true;
    for (const state of active) {
      const sampler = samplers.get(state.id)!;
      try {
        if (!first && cfg.idleMs > 0) await sleep(cfg.idleMs);
        first = false;
        if (!calibrated) await sampler.calibrate();
        const ms = await sampler.sample();
        if (!Number.isFinite(ms) || ms <= 0) {
          throw new Error(
            `measured a per-op time of ${ms}ms — the GPU pass likely did no work (check for an 'uncapturederror' in the console).`,
          );
        }
        const kept = recordSample(state, ms, cfg);
        roundStates.push(state);
        roundThrottled.push(!kept);
      } catch (error) {
        state.error = error;
      }
      await onUpdate(snapshot(state));
    }
    calibrated = true;

    if (roundStates.length > 0 && roundIsThrottled(roundThrottled, cfg)) {
      const throttledIds = roundStates.filter((_, i) => roundThrottled[i]).map((s) => s.id);
      if (cooldowns >= cfg.maxCooldowns) {
        onProgress({ type: 'throttle-abort', throttledIds });
        for (const s of states.values()) {
          if (isActive(s)) {
            s.stopReason = s.timesMs.length > 0 ? 'throttled' : undefined;
            if (s.timesMs.length === 0) {
              s.error = new Error('Every measurement was discarded as throttled and the cooldown budget is exhausted.');
            }
            await onUpdate(snapshot(s));
          }
        }
        break;
      }
      cooldowns += 1;
      onProgress({
        type: 'cooldown',
        attempt: cooldowns,
        maxAttempts: cfg.maxCooldowns,
        ms: cfg.cooldownMs,
        throttledIds,
      });
      await sleep(cfg.cooldownMs);
    }
  }

  return new Map([...states].map(([id, s]) => [id, snapshot(s)]));
}

export function resolveSamplingConfig(cfg: SamplingConfig): Required<SamplingConfig> {
  const minRounds = Math.max(1, cfg.minRounds ?? DEFAULT_SAMPLING.minRounds);
  return {
    minRounds,
    maxRounds: Math.max(minRounds, cfg.maxRounds ?? DEFAULT_SAMPLING.maxRounds),
    stableRounds: Math.max(1, cfg.stableRounds ?? DEFAULT_SAMPLING.stableRounds),
    improvementTolerance: cfg.improvementTolerance ?? DEFAULT_SAMPLING.improvementTolerance,
    throttleThreshold: cfg.throttleThreshold ?? DEFAULT_SAMPLING.throttleThreshold,
    idleMs: cfg.idleMs ?? DEFAULT_SAMPLING.idleMs,
    cooldownMs: cfg.cooldownMs ?? DEFAULT_SAMPLING.cooldownMs,
    maxCooldowns: cfg.maxCooldowns ?? DEFAULT_SAMPLING.maxCooldowns,
    throttledFraction: cfg.throttledFraction ?? DEFAULT_SAMPLING.throttledFraction,
  };
}
