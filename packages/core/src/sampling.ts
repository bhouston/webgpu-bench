import { computeStats, isBestStable, isThrottled } from './stats.ts';
import type { SamplingStopReason, Stats, SuiteProgressEvent } from './types.ts';

/** Knobs for the best-of-N sampling scheduler. */
export interface SamplingConfig {
  /** Complete kernels in catalog order (default), or interleave them for comparisons. */
  samplingOrder?: 'sequential' | 'round-robin';
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
  /** Suite-wide cooldown budget. Default 3. */
  maxCooldowns?: number;
  /** Round-robin only: throttled fraction (and at least two kernels) that triggers cooldown. Default 0.5. */
  throttledFraction?: number;
}

export const DEFAULT_SAMPLING: Required<SamplingConfig> = {
  samplingOrder: 'sequential',
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
  /** Timing budget hints for the early progress estimate; not used to time GPU work. */
  targetMs?: number;
  warmups?: number;
  /** Called after every measurement (and when a benchmark is retired) with that benchmark's state. */
  onUpdate?: (state: SampleStateSnapshot) => void | Promise<void>;
  onProgress?: (event: SuiteProgressEvent) => void;
  /** Injectable for tests; defaults to a real `setTimeout` sleep. */
  sleep?: (ms: number) => Promise<void>;
  /** Injectable wall clock for progress telemetry; does not affect GPU measurements. */
  now?: () => number;
}

const isActive = (s: SampleState) => s.stopReason === undefined && s.error === undefined;

/**
 * The idle gap doubles as the UI's render window: after the timer, wait for
 * the main thread to go idle (table re-render committed, its paint queued)
 * so the compositor's GPU work lands here rather than inside the next
 * measurement. Falls back to a plain timer where `requestIdleCallback` is
 * missing; the timeout bounds the wait on a busy page.
 */
const realSleep = (ms: number) =>
  new Promise<void>((resolve) =>
    setTimeout(() => {
      if (typeof requestIdleCallback === 'function') requestIdleCallback(() => resolve(), { timeout: 1000 });
      else resolve();
    }, ms),
  );

/**
 * A backgrounded tab gets a throttled event loop and a lower-priority GPU
 * queue, so nothing measured while hidden is a property of the device.
 * `waitVisible` blocks until the page is showing; `hiddenSince` reports
 * whether it went hidden at any point after the last `mark()`.
 */
function visibilityGuard(onPause: () => void) {
  const doc = typeof document === 'undefined' ? undefined : document;
  let hidden = false;
  const onChange = () => {
    if (doc?.hidden) {
      hidden = true;
      onPause();
    }
  };
  doc?.addEventListener('visibilitychange', onChange);
  return {
    mark: () => {
      hidden = Boolean(doc?.hidden);
    },
    hiddenSince: () => hidden || Boolean(doc?.hidden),
    waitVisible: () =>
      new Promise<void>((resolve) => {
        if (!doc?.hidden) return resolve();
        const onVisible = () => {
          if (doc.hidden) return;
          doc.removeEventListener('visibilitychange', onVisible);
          resolve();
        };
        doc.addEventListener('visibilitychange', onVisible);
      }),
    dispose: () => doc?.removeEventListener('visibilitychange', onChange),
  };
}

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

/** In-place Fisher-Yates. */
function shuffle<T>(items: T[]): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [items[i], items[j]] = [items[j]!, items[i]!];
  }
  return items;
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
 * Adaptive best-of-N sampling. Sequential mode completes each benchmark in
 * input order, with an idle gap between samples and cooldown after two
 * consecutive discards. The cooldown budget is shared by the suite; exhaustion
 * retires the noisy benchmark and continues with untouched ones.
 *
 * Optional round-robin mode shuffles active benchmarks each round and uses
 * cross-kernel agreement to trigger suite-wide cooldowns. Both modes preserve
 * calibration, minimum/maximum kept samples, best-stability stopping and the
 * discarded-attempt cap. Hidden-page measurements are dropped in either mode.
 */
export async function runSampling(
  benchmarks: readonly Sampleable[],
  options: RunSamplingOptions = {},
): Promise<Map<string, SampleStateSnapshot>> {
  const cfg = resolveSamplingConfig(options);
  const sleep = options.sleep ?? realSleep;
  const now = options.now ?? (() => performance.now());
  const onUpdate = options.onUpdate ?? (() => {});
  const onProgress = options.onProgress ?? (() => {});

  const states = new Map<string, SampleState>();
  const samplers = new Map<string, Sampleable>();
  for (const b of benchmarks) {
    states.set(b.id, { id: b.id, timesMs: [], throttledMs: [], bestMs: Number.POSITIVE_INFINITY });
    samplers.set(b.id, b);
  }
  const visibility = visibilityGuard(() => onProgress({ type: 'pause' }));
  let cooldowns = 0;
  const calibrated = new Set<string>();
  const sequential = cfg.samplingOrder === 'sequential';
  let currentId: string | undefined;
  let sampled = false;
  let consecutiveDiscards = 0;
  try {
    for (let round = 1; ; round++) {
      // Sequential mode gives early final results; round-robin randomizes
      // exposure to device temperature and remains available for comparisons.
      const pending = [...states.values()].filter(isActive);
      const active = sequential ? pending.slice(0, 1) : shuffle(pending);
      if (active.length === 0) break;
      if (sequential && currentId !== active[0]!.id) {
        currentId = active[0]!.id;
        consecutiveDiscards = 0;
        onProgress({ type: 'benchmark-start', id: currentId });
      }
      // Typical remaining measurements per still-active benchmark: assumes it
      // converges around minRounds + stableRounds, like any other. Floored at
      // 1 so a benchmark that's run longer than typical still counts as
      // "still going" rather than going negative.
      const estimatedRemainingUnits = active.reduce(
        (sum, s) => sum + Math.max(cfg.minRounds + cfg.stableRounds - s.timesMs.length, 1),
        0,
      );
      onProgress({
        type: 'round',
        round,
        active: active.length,
        estimatedRemainingUnits,
        activeIds: active.map((s) => s.id),
        sampling: cfg,
        sampleBudgetMs: options.targetMs ?? 100,
        calibrationBudgetMs: (options.targetMs ?? 100) * (2 + (options.warmups ?? 1)),
      });

      const roundStates: SampleState[] = [];
      const roundThrottled: boolean[] = [];
      let first = true;
      for (const state of active) {
        const sampler = samplers.get(state.id)!;
        let durationMs = 0;
        try {
          if (sequential && !calibrated.has(state.id)) {
            await visibility.waitVisible();
            await sampler.calibrate();
            calibrated.add(state.id);
          }
          if ((!first || (sequential && sampled)) && cfg.idleMs > 0) await sleep(cfg.idleMs);
          first = false;
          await visibility.waitVisible();
          visibility.mark();
          if (!calibrated.has(state.id)) {
            await sampler.calibrate();
            calibrated.add(state.id);
          }
          sampled = true;
          const sampleStart = now();
          const ms = await sampler.sample();
          durationMs = now() - sampleStart;
          // Hidden at any point during the measurement: not a device number. Drop it and retry next round.
          if (visibility.hiddenSince()) continue;
          if (!Number.isFinite(ms) || ms <= 0) {
            throw new Error(
              `measured a per-op time of ${ms}ms — the GPU pass likely did no work (check for an 'uncapturederror' in the console).`,
            );
          }
          const kept = recordSample(state, ms, cfg);
          if (sequential) consecutiveDiscards = kept ? 0 : consecutiveDiscards + 1;
          roundStates.push(state);
          roundThrottled.push(!kept);
        } catch (error) {
          state.error = error;
        }
        onProgress({
          type: 'sample',
          id: state.id,
          durationMs,
          timesMs: [...state.timesMs],
          throttledMs: [...state.throttledMs],
          done: !isActive(state),
        });
        await onUpdate(snapshot(state));
      }

      if (
        roundStates.length > 0 &&
        (sequential ? consecutiveDiscards >= 2 && isActive(active[0]!) : roundIsThrottled(roundThrottled, cfg))
      ) {
        const throttledIds = roundStates.filter((_, i) => roundThrottled[i]).map((s) => s.id);
        if (cooldowns >= cfg.maxCooldowns) {
          onProgress({ type: 'throttle-abort', throttledIds });
          for (const s of sequential ? active : states.values()) {
            if (isActive(s)) {
              s.stopReason = s.timesMs.length > 0 ? 'throttled' : undefined;
              if (s.timesMs.length === 0) {
                s.error = new Error(
                  'Every measurement was discarded as throttled and the cooldown budget is exhausted.',
                );
              }
              await onUpdate(snapshot(s));
            }
          }
          if (sequential) continue;
          break;
        }
        cooldowns += 1;
        consecutiveDiscards = 0;
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
  } finally {
    visibility.dispose();
  }

  return new Map([...states].map(([id, s]) => [id, snapshot(s)]));
}

export function resolveSamplingConfig(cfg: SamplingConfig): Required<SamplingConfig> {
  const minRounds = Math.max(1, cfg.minRounds ?? DEFAULT_SAMPLING.minRounds);
  return {
    samplingOrder: cfg.samplingOrder ?? DEFAULT_SAMPLING.samplingOrder,
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
