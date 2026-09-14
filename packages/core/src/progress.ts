import { DEFAULT_SAMPLING } from './sampling.ts';
import type { BenchmarkResult, SuiteProgressEvent } from './types.ts';

/** Wall-time progress with an early budget estimate refined by observed work. */
export class SuiteProgress {
  completedUnits = 0;
  private readonly terminalIds = new Set<string>();
  remainingUnits: number;
  private readonly startTime: number;
  private finished = false;
  private cfg = DEFAULT_SAMPLING;
  private paused = false;
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
    this.startTime = now();
  }

  onProgress = (event: SuiteProgressEvent): void => {
    if (event.type === 'benchmark-start') {
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
      this.cfg = event.sampling ?? DEFAULT_SAMPLING;
      this.sampleBudgetMs = event.sampleBudgetMs ?? 100;
      this.calibrationBudgetMs = event.calibrationBudgetMs ?? 300;
    } else if (event.type === 'sample') {
      this.paused = false;
      this.completedUnits++;
      this.remainingUnits = Math.max(0, this.remainingUnits - 1);
      if (event.done) this.recordCompletion(event.id);
    }
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

  private estimateMs(mode: 'expected' | 'lower' | 'upper', at: number): number | null {
    if (
      this.finished ||
      this.paused ||
      this.benchmarkCount === 0 ||
      (typeof document !== 'undefined' && document.hidden)
    )
      return null;
    const remaining = this.sequentialRemainingMs(at);
    if (mode === 'expected') return remaining;
    const durations = [...this.benchmarkDurations.values()];
    const mean = durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : 1;
    const spread = durations.length < 2 ? 0.5 : Math.max(0.1, ...durations.map((v) => Math.abs(v / mean - 1)));
    const start = this.currentId ? this.benchmarkStarts.get(this.currentId) : undefined;
    const age = start === undefined ? 0 : at - start - (this.benchmarkCooldowns.get(this.currentId!) ?? 0);
    const overrun =
      durations.length && this.currentId && !this.benchmarkDurations.has(this.currentId) ? Math.max(0, age - mean) : 0;
    return mode === 'lower' ? Math.max(0, remaining * (1 - spread)) : remaining * (1 + spread) + overrun;
  }

  /** Whether current uncertainty supports a roughly 20%-accurate ETA. Rough estimates remain visible. */
  get isApproximate(): boolean {
    const at = this.now();
    const remaining = this.estimateMs('expected', at);
    if (remaining === null || this.round <= this.blockedUntilRound || at < this.cooldownUntil) return true;
    if (this.currentId && !this.benchmarkDurations.has(this.currentId)) {
      const durations = [...this.benchmarkDurations.values()];
      const mean = durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : Infinity;
      const age =
        at - (this.benchmarkStarts.get(this.currentId) ?? at) - (this.benchmarkCooldowns.get(this.currentId) ?? 0);
      if (age > mean * 1.2) return true;
    }
    const lower = this.estimateMs('lower', at)!;
    const upper = this.estimateMs('upper', at)!;
    return (
      this.benchmarkDurations.size < 2 || lower <= 0 || Math.max(remaining / lower - 1, 1 - remaining / upper) > 0.2
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
