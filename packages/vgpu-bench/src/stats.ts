import type { Stats } from './types.ts';

/**
 * Two-sided 97.5% quantiles of Student's t distribution (i.e. the multiplier
 * for a 95% confidence interval) indexed by degrees of freedom 1..30. Beyond
 * 30 the value is within ~4% of the normal 1.96, so the df=30 entry is reused.
 */
const T_975 = [
  12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131, 2.12, 2.11,
  2.101, 2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048, 2.045, 2.042,
];

/** Student-t multiplier for a two-sided 95% confidence interval with `df` degrees of freedom. */
export function tQuantile975(df: number): number {
  if (df < 1) return Number.POSITIVE_INFINITY;
  return T_975[Math.min(df, T_975.length) - 1]!;
}

export function computeStats(values: readonly number[]): Stats {
  if (values.length === 0) {
    throw new Error('computeStats: values must be non-empty');
  }
  const n = values.length;
  const sorted = values.toSorted((a, b) => a - b);
  const sum = values.reduce((a, b) => a + b, 0);
  const mean = sum / n;
  const min = sorted[0]!;
  const max = sorted[n - 1]!;
  const mid = Math.floor(n / 2);
  const median = n % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
  const variance = n > 1 ? values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (n - 1) : 0;
  const stddev = Math.sqrt(variance);
  // Half-width of the 95% CI on the mean. With a single sample the
  // t-multiplier is infinite (df = 0), so the interval is unbounded.
  const ci95 = n > 1 ? (tQuantile975(n - 1) * stddev) / Math.sqrt(n) : Number.POSITIVE_INFINITY;
  return { mean, min, max, median, stddev, ci95 };
}

export interface BestStableOptions {
  /** Never declare the best stable with fewer samples than this. */
  minRuns: number;
  /** The best must have gone this many consecutive samples without a meaningful improvement. */
  stableRuns: number;
  /** A new best more than this fraction below the previous best counts as an improvement (0.01 = 1%). */
  tolerance: number;
}

/**
 * Stopping rule for best-of-N sampling.
 *
 * The reported statistic is the *minimum* time, since every source of noise a
 * benchmark meets (thermal throttling, clock ramp, compositor frames, other
 * apps) only ever makes a run slower — the fastest run is the closest thing
 * to the device's true capability. So sampling is done once the minimum has
 * settled: the last `stableRuns` samples failed to beat the best seen before
 * them by more than `tolerance`. Three tight samples converge immediately
 * (minRuns 3, stableRuns 2: sample 1 sets the best, samples 2 and 3 confirm
 * it); a still-ramping GPU that keeps producing faster runs keeps sampling.
 *
 * Degenerate data (non-positive or non-finite best) never converges, so the
 * caller runs to its cap and can then diagnose the bad timings itself.
 */
export function isBestStable(values: readonly number[], opts: BestStableOptions): boolean {
  const n = values.length;
  if (n < Math.max(opts.stableRuns + 1, opts.minRuns, 2)) return false;
  const best = Math.min(...values);
  if (!Number.isFinite(best) || best <= 0) return false;
  const priorBest = Math.min(...values.slice(0, n - opts.stableRuns));
  // Stable iff none of the trailing samples improved on the prior best by more than the tolerance.
  return best >= priorBest * (1 - opts.tolerance);
}

/** True when `sampleMs` is more than `threshold` (fraction) slower than `bestMs` — the signature of a throttled run. */
export function isThrottled(sampleMs: number, bestMs: number, threshold: number): boolean {
  if (!Number.isFinite(bestMs) || bestMs <= 0) return false;
  return sampleMs > bestMs * (1 + threshold);
}
