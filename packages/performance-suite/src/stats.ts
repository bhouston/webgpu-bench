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

export interface ConvergenceOptions {
  /** Never declare convergence with fewer samples than this. */
  minRuns: number;
  /** Target: 95% CI half-width on the mean, as a fraction of the mean (0.03 = ±3%). */
  precision: number;
}

/**
 * Sequential stopping rule for adaptive sampling.
 *
 * After each measurement the harness asks whether the samples gathered so far
 * already pin down the mean tightly enough. "Tightly enough" is defined via a
 * Student-t confidence interval: with `n` samples of standard deviation `s`,
 * the true mean lies within `t(n-1) * s / sqrt(n)` of the sample mean with 95%
 * confidence. Convergence means that half-width is at most `precision` times
 * the mean, so e.g. 3 samples with a coefficient of variation below ~1.2%
 * converge at ±3%, while noisier data needs more samples before the shrinking
 * `t / sqrt(n)` factor gets the interval under the target.
 *
 * Because the rule is evaluated repeatedly ("optional stopping") the realised
 * confidence is somewhat below the nominal 95% — acceptable for a benchmark
 * whose reported statistic is the median anyway, and bounded by `minRuns`.
 *
 * Degenerate data (non-positive or non-finite mean) never converges, so the
 * caller runs to its cap and can then diagnose the bad timings itself.
 */
export function hasConverged(values: readonly number[], opts: ConvergenceOptions): boolean {
  if (values.length < Math.max(2, opts.minRuns)) return false;
  const { mean, ci95 } = computeStats(values);
  if (!Number.isFinite(mean) || mean <= 0) return false;
  return ci95 / mean <= opts.precision;
}
