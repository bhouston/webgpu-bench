import { DEFAULT_SAMPLING } from './sampling.ts';
import { isBestStable } from './stats.ts';
import type { BenchmarkResult, SuiteProgressEvent } from './types.ts';

/** Wall-time progress; displayFraction is null while the estimate lacks evidence. */
export class SuiteProgress {
  completedUnits = 0;
  remainingUnits: number;
  private readonly startTime: number;
  private updatedAt: number;
  private finished = false;
  private activeIds: string[] = [];
  private pending = new Set<string>();
  private samples = new Map<string, { durations: number[]; timesMs: number[]; throttledMs: number[]; done: boolean }>();
  private roundSampled = false;
  private cfg = DEFAULT_SAMPLING;
  private calibrated = false;
  private round = 0;
  private blockedUntilRound = 0;

  constructor(
    benchmarkCount: number,
    private readonly now: () => number = () => performance.now(),
  ) {
    this.remainingUnits = benchmarkCount * (DEFAULT_SAMPLING.minRounds + DEFAULT_SAMPLING.stableRounds);
    this.startTime = this.updatedAt = now();
  }

  onProgress = (event: SuiteProgressEvent): void => {
    if (event.type === 'cooldown' || event.type === 'pause' || event.type === 'throttle-abort') {
      this.blockedUntilRound = this.round + 2;
    }
    if (event.type === 'round') {
      this.round = event.round;
      this.remainingUnits = event.estimatedRemainingUnits;
      this.activeIds = event.activeIds ?? [];
      this.pending = new Set(this.activeIds);
      this.roundSampled = false;
      this.cfg = event.sampling ?? DEFAULT_SAMPLING;
    } else if (event.type === 'sample') {
      this.completedUnits++;
      this.remainingUnits = Math.max(0, this.remainingUnits - 1);
      this.pending.delete(event.id);
      this.roundSampled = true;
      const previous = this.samples.get(event.id);
      const durations = previous?.durations ?? [];
      const lastDuration = durations.at(-1);
      if (
        event.throttledMs.length > (previous?.throttledMs.length ?? 0) ||
        !(event.durationMs > 0) ||
        !Number.isFinite(event.durationMs) ||
        (lastDuration !== undefined && Math.abs(event.durationMs / lastDuration - 1) > 0.2)
      ) {
        this.blockedUntilRound = this.round + 2;
      }
      if (event.durationMs > 0 && Number.isFinite(event.durationMs)) durations.push(event.durationMs);
      this.samples.set(event.id, {
        durations,
        timesMs: event.timesMs,
        throttledMs: event.throttledMs,
        done: event.done,
      });
      if (this.pending.size === 0) this.calibrated = true;
    }
    this.updatedAt = this.now();
  };

  /** Rows populate result tables; scheduler telemetry accounts for work exactly once. */
  onResult = (_result: BenchmarkResult): void => {};

  finish(): void {
    this.finished = true;
    this.remainingUnits = 0;
  }

  private minimumSamples(times: readonly number[]): number {
    const future = [...times];
    const best = Math.min(...times);
    while (
      future.length < this.cfg.maxRounds &&
      !isBestStable(future, {
        minRuns: this.cfg.minRounds,
        stableRuns: this.cfg.stableRounds,
        tolerance: this.cfg.improvementTolerance,
      })
    )
      future.push(best);
    return Math.max(1, future.length - times.length);
  }

  private predictSamples(times: readonly number[]): number {
    const minimum = this.minimumSamples(times);
    const cap = Math.max(1, this.cfg.maxRounds - times.length);
    if (minimum >= cap) return cap;
    let improvements = 0,
      best = times[0] ?? Infinity;
    for (const value of times.slice(1)) {
      if (value < best * (1 - this.cfg.improvementTolerance)) improvements++;
      best = Math.min(best, value);
    }
    // Small quiet prior; adapt to this benchmark's observed improvement frequency.
    const probability = (improvements + 0.1) / (Math.max(0, times.length - 1) + 2);
    let streak = 0;
    for (let i = times.length - 1; i > 0; i--) {
      const prior = Math.min(...times.slice(0, i));
      if (times[i]! < prior * (1 - this.cfg.improvementTolerance)) break;
      streak++;
    }
    let distribution = new Map([[Math.min(streak, this.cfg.stableRounds), 1]]);
    let expected = 0;
    for (let step = 1; step <= cap; step++) {
      expected += [...distribution.values()].reduce((a, b) => a + b, 0);
      const next = new Map<number, number>();
      for (const [stable, mass] of distribution) {
        next.set(0, (next.get(0) ?? 0) + mass * probability);
        const nextStable = Math.min(this.cfg.stableRounds, stable + 1);
        if (times.length + step < Math.max(2, this.cfg.minRounds) || nextStable < this.cfg.stableRounds)
          next.set(nextStable, (next.get(nextStable) ?? 0) + mass * (1 - probability));
      }
      distribution = next;
    }
    return Math.max(minimum, expected);
  }

  private estimateMs(mode: 'expected' | 'lower' | 'upper'): number | null {
    if (!this.calibrated || this.finished || this.round < this.blockedUntilRound) return null;
    const nextId = this.pending.values().next().value;
    const nextDuration = nextId ? this.samples.get(nextId)?.durations.at(-1) : undefined;
    if (this.now() - this.updatedAt > Math.max(250, ((nextDuration ?? 0) + this.cfg.idleMs) * 1.5)) return null;
    let estimate = 0,
      futureSamples = 0,
      futureRounds = 0,
      currentSamples = 0;
    if (mode === 'expected') this.remainingUnits = 0;
    for (const [id, state] of this.samples) {
      if (state.done) continue;
      if (state.durations.length < 2) return null;
      const cap = Math.max(1, this.cfg.maxRounds - state.timesMs.length);
      const expected = this.predictSamples(state.timesMs);
      // Empirical envelope: allow another full stability window of improvements.
      // It is intentionally not a confidence interval or a guarantee of future GPU behavior.
      const count =
        mode === 'lower'
          ? this.minimumSamples(state.timesMs)
          : mode === 'upper'
            ? Math.min(cap, expected + this.cfg.stableRounds)
            : expected;
      if (mode === 'expected') this.remainingUnits += count;
      const mean = state.durations.reduce((a, b) => a + b, 0) / state.durations.length;
      const duration =
        mode === 'lower'
          ? Math.min(...state.durations.slice(-3)) * 0.9
          : mode === 'upper'
            ? Math.max(...state.durations.slice(-3)) * 1.1
            : mean;
      estimate += duration * count;
      const current = this.pending.has(id) ? 1 : 0;
      currentSamples += current;
      futureSamples += count - current;
      futureRounds = Math.max(futureRounds, count - current);
    }
    const gaps = Math.max(0, currentSamples - (this.roundSampled ? 0 : 1)) + futureSamples - futureRounds;
    estimate += this.cfg.idleMs * gaps;
    return Math.max(0, estimate - (this.now() - this.updatedAt));
  }

  /** @deprecated Use displayFraction; numeric compatibility returns 0 while indeterminate. */
  get fraction(): number {
    return this.displayFraction ?? 0;
  }

  /** Null means show an indeterminate status, not a percentage or a filled bar. */
  get displayFraction(): number | null {
    if (this.finished) return 1;
    const remaining = this.estimateMs('expected');
    if (remaining === null) return null;
    const elapsed = this.now() - this.startTime;
    const lower = this.estimateMs('lower')!;
    const upper = this.estimateMs('upper')!;
    if (Math.max(remaining - lower, upper - remaining) / (elapsed + remaining) > 0.2) return null;
    return Math.min(0.999, elapsed / Math.max(1, elapsed + remaining));
  }

  /** Unrounded seconds left; separately gated because an accurate percentage need not imply an accurate ETA. */
  get remainingSeconds(): number | null {
    const remaining = this.estimateMs('expected');
    if (remaining === null) return null;
    const lower = this.estimateMs('lower')!;
    const upper = this.estimateMs('upper')!;
    if (lower <= 0 || Math.max(remaining / lower - 1, 1 - remaining / upper) > 0.2) return null;
    return remaining / 1000;
  }
}
