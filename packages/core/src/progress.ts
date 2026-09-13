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

  constructor(
    benchmarkCount: number,
    private readonly now: () => number = () => performance.now(),
  ) {
    this.remainingUnits = benchmarkCount * (DEFAULT_SAMPLING.minRounds + DEFAULT_SAMPLING.stableRounds);
    this.startTime = this.updatedAt = now();
  }

  onProgress = (event: SuiteProgressEvent): void => {
    if (event.type === 'round') {
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

  private get remainingMs(): number | null {
    if (!this.calibrated || this.finished) return null;
    let estimate = 0,
      futureSamples = 0,
      futureRounds = 0,
      currentSamples = 0;
    this.remainingUnits = 0;
    for (const [id, state] of this.samples) {
      if (state.done) continue;
      if (state.durations.length === 0) return null;
      const count = this.predictSamples(state.timesMs);
      this.remainingUnits += count;
      const mean = state.durations.reduce((a, b) => a + b, 0) / state.durations.length;
      estimate += mean * count;
      const current = this.pending.has(id) ? 1 : 0;
      currentSamples += current;
      futureSamples += count - current;
      futureRounds = Math.max(futureRounds, count - current);
    }
    const gaps = Math.max(0, currentSamples - (this.roundSampled ? 0 : 1)) + futureSamples - futureRounds;
    estimate += this.cfg.idleMs * gaps;
    return Math.max(0, estimate - (this.now() - this.updatedAt));
  }

  /** Numeric compatibility accessor. Use displayFraction to render a trustworthy percentage. */
  get fraction(): number {
    return this.displayFraction ?? 0;
  }

  /** Null means show an indeterminate status, not a percentage or a filled bar. */
  get displayFraction(): number | null {
    if (this.finished) return 1;
    const remaining = this.remainingMs;
    if (remaining === null) return null;
    const elapsed = this.now() - this.startTime;
    return Math.min(0.999, elapsed / Math.max(1, elapsed + remaining));
  }

  /** Projected seconds left; null when there is insufficient timing evidence. */
  get remainingSeconds(): number | null {
    const remaining = this.remainingMs;
    return remaining === null ? null : remaining / 1000;
  }
}
