import { computeStats, isBestStable, isThrottled } from './stats.ts';
import type { SamplingStopReason, Stats, SuiteProgressEvent } from './types.ts';

/** Knobs for the best-of-N sampling scheduler. */
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
  /** Suite-wide cooldown budget. Default 3. */
  maxCooldowns?: number;
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
    // Bound discarded attempts even when a generous cooldown budget allows
    // a persistently slow benchmark to keep retrying.
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

function snapshot(state: SampleState): SampleStateSnapshot {
  return {
    ...state,
    timesMs: [...state.timesMs],
    throttledMs: [...state.throttledMs],
    stats: state.timesMs.length > 0 ? computeStats(state.timesMs) : undefined,
  };
}

/**
 * Complete benchmarks in input order, resting between measurements. Two
 * consecutive discards trigger a cooldown. The suite shares a finite cooldown
 * budget; exhaustion retires the noisy benchmark and continues with later ones.
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
  const results = new Map<string, SampleStateSnapshot>();
  const visibility = visibilityGuard(() => onProgress({ type: 'pause' }));
  let cooldowns = 0;
  let sampled = false;
  let round = 0;
  try {
    for (const benchmark of benchmarks) {
      const state: SampleState = { id: benchmark.id, timesMs: [], throttledMs: [], bestMs: Infinity };
      let calibrated = false;
      let consecutiveDiscards = 0;
      onProgress({ type: 'benchmark-start', id: state.id });
      while (isActive(state)) {
        onProgress({
          type: 'round',
          round: ++round,
          active: 1,
          estimatedRemainingUnits: Math.max(cfg.minRounds + cfg.stableRounds - state.timesMs.length, 1),
          activeIds: [state.id],
          sampling: cfg,
          sampleBudgetMs: options.targetMs ?? 100,
          calibrationBudgetMs: (options.targetMs ?? 100) * (2 + (options.warmups ?? 1)),
        });
        let durationMs = 0;
        try {
          if (!calibrated) {
            await visibility.waitVisible();
            await benchmark.calibrate();
            calibrated = true;
          }
          if (sampled && cfg.idleMs > 0) await sleep(cfg.idleMs);
          await visibility.waitVisible();
          visibility.mark();
          sampled = true;
          const sampleStart = now();
          const ms = await benchmark.sample();
          durationMs = now() - sampleStart;
          // Hidden-page measurements do not represent device performance.
          if (visibility.hiddenSince()) continue;
          if (!Number.isFinite(ms) || ms <= 0) {
            throw new Error(
              `measured a per-op time of ${ms}ms — the GPU pass likely did no work (check for an 'uncapturederror' in the console).`,
            );
          }
          consecutiveDiscards = recordSample(state, ms, cfg) ? 0 : consecutiveDiscards + 1;
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
        if (consecutiveDiscards >= 2 && isActive(state)) {
          const throttledIds = [state.id];
          if (cooldowns >= cfg.maxCooldowns) {
            onProgress({ type: 'throttle-abort', throttledIds });
            state.stopReason = 'throttled';
            await onUpdate(snapshot(state));
            break;
          }
          consecutiveDiscards = 0;
          onProgress({
            type: 'cooldown',
            attempt: ++cooldowns,
            maxAttempts: cfg.maxCooldowns,
            ms: cfg.cooldownMs,
            throttledIds,
          });
          await sleep(cfg.cooldownMs);
        }
      }
      results.set(state.id, snapshot(state));
    }
  } finally {
    visibility.dispose();
  }
  return results;
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
  };
}
