import { DEFAULT_SAMPLING } from './sampling.ts';
import { isBestStable } from './stats.ts';
import type { BenchmarkResult, SuiteProgressEvent } from './types.ts';

/** Wall-time progress with an early budget estimate refined by observed work. */
export class SuiteProgress {
  completedUnits = 0;
  private readonly terminalIds = new Set<string>();
  remainingUnits: number;
  private readonly startTime: number;
  private updatedAt: number;
  private finished = false;
  private activeIds: string[] = [];
  private pending = new Set<string>();
  private samples = new Map<
    string,
    { durations: number[]; timesMs: number[]; throttledMs: number[]; done: boolean; unreliableUntil: number }
  >();
  private roundSampled = false;
  private cfg = DEFAULT_SAMPLING;
  private readonly calibrations = new Map<string, number>();
  private paused = false;
  private order: 'sequential' | 'round-robin' = 'sequential';
  private sampleBudgetMs = 100;
  private calibrationBudgetMs = 300;
  private readonly benchmarkStarts = new Map<string, number>();
  private readonly benchmarkDurations = new Map<string, number>();
  private readonly benchmarkCooldowns = new Map<string, number>();
  private currentId: string | undefined;
  private cooldownUntil = 0;
  private cooldownsUsed = 0;
  private round = 0;
  private blockedUntilRound = 0;

  constructor(
    readonly benchmarkCount: number,
    private readonly now: () => number = () => performance.now(),
  ) {
    this.remainingUnits = benchmarkCount * (DEFAULT_SAMPLING.minRounds + DEFAULT_SAMPLING.stableRounds);
    this.startTime = this.updatedAt = now();
  }

  onProgress = (event: SuiteProgressEvent): void => {
    if (event.type === 'benchmark-start') {
      this.order = 'sequential';
      this.currentId = event.id;
      this.benchmarkStarts.set(event.id, this.now());
      this.paused = false;
      this.cooldownUntil = 0;
    }
    if (event.type === 'cooldown' || event.type === 'pause' || event.type === 'throttle-abort') {
      this.blockedUntilRound = this.round + 2;
    }
    if (event.type === 'pause' || event.type === 'throttle-abort') this.paused = true;
    if (event.type === 'cooldown') {
      this.cooldownsUsed = event.attempt;
      this.cooldownUntil = this.now() + event.ms;
      if (this.currentId)
        this.benchmarkCooldowns.set(this.currentId, (this.benchmarkCooldowns.get(this.currentId) ?? 0) + event.ms);
    }
    if (event.type === 'round') {
      this.paused = false;
      this.round = event.round;
      this.remainingUnits = event.estimatedRemainingUnits;
      this.activeIds = event.activeIds ?? [];
      this.pending = new Set(this.activeIds);
      this.roundSampled = false;
      this.cfg = event.sampling ?? DEFAULT_SAMPLING;
      // Recorded/older schedulers without an order field used round-robin.
      this.order = event.sampling?.samplingOrder ?? 'round-robin';
      this.sampleBudgetMs = event.sampleBudgetMs ?? 100;
      this.calibrationBudgetMs = event.calibrationBudgetMs ?? 300;
    } else if (event.type === 'sample') {
      this.paused = false;
      this.completedUnits++;
      this.remainingUnits = Math.max(0, this.remainingUnits - 1);
      this.pending.delete(event.id);
      const previous = this.samples.get(event.id);
      if (!previous && Number.isFinite(event.durationMs) && event.durationMs > 0) {
        // Older traces lack calibration telemetry. Infer its wall cost from the
        // first sample interval, excluding the sample and the preceding idle gap.
        const idle = this.roundSampled ? this.cfg.idleMs : 0;
        this.calibrations.set(event.id, Math.max(0, this.now() - this.updatedAt - event.durationMs - idle));
      }
      this.roundSampled = true;
      const durations = previous?.durations ?? [];
      const lastDuration = durations.at(-1);
      let unreliableUntil = previous?.unreliableUntil ?? 0;
      if (
        event.throttledMs.length > (previous?.throttledMs.length ?? 0) ||
        !(event.durationMs > 0) ||
        !Number.isFinite(event.durationMs) ||
        (lastDuration !== undefined && Math.abs(event.durationMs / lastDuration - 1) > 0.2)
      ) {
        unreliableUntil = this.round + 2;
      }
      if (event.durationMs > 0 && Number.isFinite(event.durationMs)) durations.push(event.durationMs);
      this.samples.set(event.id, {
        durations,
        timesMs: event.timesMs,
        throttledMs: event.throttledMs,
        done: event.done,
        unreliableUntil,
      });
      if (event.done) this.recordCompletion(event.id);
    }
    this.updatedAt = this.now();
  };

  /** Exact completed-test count, independent of any runtime estimate. */
  get completedBenchmarks(): number {
    return this.terminalIds.size;
  }

  /** Terminal rows advance the exact count once; sampling telemetry predicts time. */
  onResult = (result: BenchmarkResult): void => {
    if (result.status === 'ok' || result.status === 'skipped' || result.status === 'error') {
      this.terminalIds.add(result.id);
      this.recordCompletion(result.id);
    }
  };

  private recordCompletion(id: string): void {
    const start = this.benchmarkStarts.get(id);
    if (start !== undefined && !this.benchmarkDurations.has(id)) {
      this.benchmarkDurations.set(id, Math.max(1, this.now() - start - (this.benchmarkCooldowns.get(id) ?? 0)));
    }
  }

  private sequentialRemainingMs(at: number): number {
    const completed = new Set([...this.terminalIds, ...this.benchmarkDurations.keys()]);
    const remainingCount = Math.max(0, this.benchmarkCount - completed.size);
    if (remainingCount === 0) return 0;
    const measured = [...this.benchmarkDurations.values()];
    const samples = Math.min(this.cfg.maxRounds, Math.max(this.cfg.minRounds, this.cfg.stableRounds + 1));
    const budget = this.calibrationBudgetMs + samples * (this.sampleBudgetMs + this.cfg.idleMs);
    const mean = measured.length ? measured.reduce((a, b) => a + b, 0) / measured.length : budget;
    const start = this.currentId ? this.benchmarkStarts.get(this.currentId) : undefined;
    const remainingCooldown = Math.max(0, this.cooldownUntil - at);
    const elapsedCooldown = this.currentId ? (this.benchmarkCooldowns.get(this.currentId) ?? 0) - remainingCooldown : 0;
    const age = start !== undefined && !completed.has(this.currentId!) ? Math.max(0, at - start - elapsedCooldown) : 0;
    // An unexpectedly slow current benchmark must not consume the budget of all
    // following benchmarks or leave the display stuck at zero seconds.
    const coolingObserved = [...this.benchmarkCooldowns.values()].reduce((a, b) => a + b, 0);
    const expectedFutureCooling = Math.min(
      Math.max(0, this.cfg.maxCooldowns - this.cooldownsUsed) * this.cfg.cooldownMs,
      (coolingObserved / Math.max(1, measured.length + 1)) * remainingCount,
    );
    return remainingCooldown + expectedFutureCooling + (remainingCount - 1) * mean + Math.max(mean * 0.15, mean - age);
  }

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

  private estimateMs(mode: 'expected' | 'lower' | 'upper', at: number): number | null {
    if (
      this.finished ||
      this.paused ||
      this.benchmarkCount === 0 ||
      (typeof document !== 'undefined' && document.hidden)
    )
      return null;
    if (this.order === 'sequential') {
      const remaining = this.sequentialRemainingMs(at);
      if (mode === 'expected') return remaining;
      const durations = [...this.benchmarkDurations.values()];
      const mean = durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : 1;
      const spread = durations.length < 2 ? 0.5 : Math.max(0.1, ...durations.map((v) => Math.abs(v / mean - 1)));
      const start = this.currentId ? this.benchmarkStarts.get(this.currentId) : undefined;
      const age = start === undefined ? 0 : at - start - (this.benchmarkCooldowns.get(this.currentId!) ?? 0);
      const overrun =
        durations.length && this.currentId && !this.benchmarkDurations.has(this.currentId)
          ? Math.max(0, age - mean)
          : 0;
      return mode === 'lower' ? Math.max(0, remaining * (1 - spread)) : remaining * (1 + spread) + overrun;
    }
    const observations = [...this.samples.values()].flatMap((s) => s.durations);
    const typicalDuration = observations.length
      ? observations.reduce((a, b) => a + b, 0) / observations.length
      : this.sampleBudgetMs;
    const calibrationTimes = [...this.calibrations.values()];
    const typicalCalibration = calibrationTimes.length
      ? calibrationTimes.reduce((a, b) => a + b, 0) / calibrationTimes.length
      : this.calibrationBudgetMs;
    const knownIds = new Set([...this.samples.keys(), ...this.terminalIds]);
    const unseen = Math.max(0, this.benchmarkCount - knownIds.size);
    const initialSamples = Math.min(this.cfg.maxRounds, 1 + this.predictSamples([1]));
    const unseenSamples =
      mode === 'lower'
        ? Math.min(this.cfg.maxRounds, Math.max(this.cfg.minRounds, this.cfg.stableRounds + 1))
        : mode === 'upper'
          ? Math.min(this.cfg.maxRounds, initialSamples + this.cfg.stableRounds)
          : initialSamples;
    const durationAllowance = mode === 'lower' ? 0.7 : mode === 'upper' ? 1.5 : 1;
    let estimate =
      unseen *
      (typicalCalibration * (mode === 'lower' ? 0.5 : mode === 'upper' ? 2 : 1) +
        typicalDuration * unseenSamples * durationAllowance);
    // During the first round unseen kernels each have one current sample left.
    let currentSamples = unseen,
      futureSamples = unseen * Math.max(0, unseenSamples - 1),
      futureRounds = unseen ? Math.max(0, unseenSamples - 1) : 0;
    let units = unseen * unseenSamples;
    const quietRetired = [...this.samples.values()].filter(
      (s) =>
        s.done &&
        s.throttledMs.length === 0 &&
        isBestStable(s.timesMs, {
          minRuns: this.cfg.minRounds,
          stableRuns: this.cfg.stableRounds,
          tolerance: this.cfg.improvementTolerance,
        }),
    ).length;
    const stabilityAllowance = this.cfg.stableRounds * Math.max(0.5, 1 - quietRetired / Math.max(1, this.samples.size));
    for (const [id, state] of this.samples) {
      if (state.done) continue;
      const cap = Math.max(1, this.cfg.maxRounds - state.timesMs.length);
      const uncertain = this.round <= state.unreliableUntil;
      const attempts = state.timesMs.length + state.throttledMs.length;
      const retryCap = Math.max(0, 2 * this.cfg.maxRounds - attempts - 1) + cap;
      const expected = Math.min(
        retryCap,
        this.predictSamples(state.timesMs) * (uncertain ? attempts / Math.max(1, state.timesMs.length) : 1),
      );
      const count =
        mode === 'lower'
          ? Math.min(this.minimumSamples(state.timesMs), Math.max(1, 2 * this.cfg.maxRounds - attempts))
          : mode === 'upper'
            ? uncertain
              ? retryCap
              : Math.min(cap, expected + stabilityAllowance)
            : expected;
      units += count;
      const durations = state.durations.length ? state.durations : [typicalDuration];
      const mean = durations.reduce((a, b) => a + b, 0) / durations.length;
      const duration =
        mode === 'lower'
          ? Math.min(...durations.slice(-3)) * 0.9
          : mode === 'upper'
            ? Math.max(...durations.slice(-3)) * 1.1
            : mean;
      estimate += duration * count;
      const current = this.pending.has(id) ? 1 : 0;
      currentSamples += current;
      futureSamples += count - current;
      futureRounds = Math.max(futureRounds, count - current);
    }
    if (mode === 'expected') this.remainingUnits = units;
    const gaps = Math.max(0, currentSamples - (this.roundSampled ? 0 : 1)) + futureSamples - futureRounds;
    estimate += this.cfg.idleMs * gaps;
    const cooldown = Math.max(0, this.cooldownUntil - at);
    // Do not let an overrun turn into a stuck zero-second countdown. Once late,
    // retain a sample-sized allowance and mark the estimate approximate.
    const age = Math.max(0, at - Math.max(this.updatedAt, this.cooldownUntil));
    const remaining = estimate - age;
    return cooldown + Math.max(remaining, units > 0 ? Math.min(estimate, typicalDuration) : 0);
  }

  /** Whether current uncertainty supports a roughly 20%-accurate ETA. Rough estimates remain visible. */
  get isApproximate(): boolean {
    const at = this.now();
    const remaining = this.estimateMs('expected', at);
    if (remaining === null || this.round <= this.blockedUntilRound || at < this.cooldownUntil) return true;
    if (this.order === 'sequential' && this.currentId && !this.benchmarkDurations.has(this.currentId)) {
      const durations = [...this.benchmarkDurations.values()];
      const mean = durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : Infinity;
      const age =
        at - (this.benchmarkStarts.get(this.currentId) ?? at) - (this.benchmarkCooldowns.get(this.currentId) ?? 0);
      if (age > mean * 1.2) return true;
    }
    const lower = this.estimateMs('lower', at)!;
    const upper = this.estimateMs('upper', at)!;
    return (
      (this.order === 'sequential' ? this.benchmarkDurations.size < 2 : this.samples.size === 0) ||
      lower <= 0 ||
      Math.max(remaining / lower - 1, 1 - remaining / upper) > 0.2
    );
  }

  /** Empirical uncertainty range in seconds; useful alongside an approximate countdown. */
  get remainingSecondsRange(): readonly [number, number] | null {
    const at = this.now();
    const lower = this.estimateMs('lower', at);
    const upper = this.estimateMs('upper', at);
    return lower === null || upper === null ? null : [lower / 1000, upper / 1000];
  }

  /** @deprecated Use displayFraction; numeric compatibility returns 0 while indeterminate. */
  get fraction(): number {
    return this.displayFraction ?? 0;
  }

  /** Estimated wall-time fraction. Null only for an empty or suspended suite. */
  get displayFraction(): number | null {
    if (this.finished) return 1;
    const at = this.now();
    const remaining = this.estimateMs('expected', at);
    if (remaining === null) return null;
    const elapsed = at - this.startTime;
    return Math.min(0.999, elapsed / Math.max(1, elapsed + remaining));
  }

  /** Unrounded seconds left. Use isApproximate to qualify estimates with limited evidence. */
  get remainingSeconds(): number | null {
    const at = this.now();
    const remaining = this.estimateMs('expected', at);
    if (remaining === null) return null;
    return remaining / 1000;
  }
}
