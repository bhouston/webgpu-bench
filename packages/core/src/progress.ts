import { DEFAULT_SAMPLING } from './sampling.ts';
import type { BenchmarkResult, SuiteProgressEvent } from './types.ts';

/**
 * Turns `runSuite`'s stream (one row per change, plus `SuiteProgressEvent`s)
 * into a completion fraction and an ETA, so every front end — a browser
 * progress bar, a CLI percentage — computes them the same way.
 *
 * Wire `onProgress` into `runSuite`'s options and call `onResult` for every
 * yielded row. One unit per yielded row (setup, every sample, the final
 * status) so the fraction advances steadily; the denominator is
 * `completed + remaining`, where `remaining` is the library's own
 * per-round estimate (`SuiteProgressEvent.estimatedRemainingUnits`) and,
 * before the first round, a typical-rounds guess.
 */
export class SuiteProgress {
  completedUnits = 0;
  remainingUnits: number;
  private measuredUnits = 0;
  private startTime = 0;

  constructor(
    private readonly benchmarkCount: number,
    private readonly now: () => number = () => performance.now(),
  ) {
    // A benchmark can converge as early as minRounds, but only after stableRounds more without improving.
    this.remainingUnits = benchmarkCount * (DEFAULT_SAMPLING.minRounds + DEFAULT_SAMPLING.stableRounds + 1);
  }

  onProgress = (event: SuiteProgressEvent): void => {
    if (event.type !== 'round') return;
    this.remainingUnits = event.estimatedRemainingUnits;
    // Setup (shader compiles) before round 1 is comparatively instant; folding
    // it into the measurement rate biases the ETA low, so the clock starts here.
    if (event.round === 1) this.startTime = this.now();
  };

  onResult = (_result: BenchmarkResult): void => {
    this.completedUnits += 1;
    if (this.completedUnits > this.benchmarkCount) this.measuredUnits += 1;
  };

  /** Call once the suite's generator has finished so `fraction` reads exactly 1. */
  finish(): void {
    this.remainingUnits = 0;
  }

  /** 0..1 */
  get fraction(): number {
    const total = this.completedUnits + this.remainingUnits;
    return total === 0 ? 0 : this.completedUnits / total;
  }

  /** Projected seconds left, or null until a few measurements have landed (one slow warmup dispatch would otherwise swing it wildly). */
  get remainingSeconds(): number | null {
    if (this.measuredUnits < 3 || this.remainingUnits === 0) return null;
    const elapsedMs = this.now() - this.startTime;
    const estimate = Math.round(((elapsedMs / this.measuredUnits) * this.remainingUnits) / 1000);
    return Number.isFinite(estimate) ? estimate : null;
  }
}
