import { describe, expect, it } from 'vitest';
import { computeStats, hasConverged, tQuantile975 } from './stats.ts';

describe('tQuantile975', () => {
  it('matches the textbook table', () => {
    expect(tQuantile975(1)).toBeCloseTo(12.706, 3);
    expect(tQuantile975(2)).toBeCloseTo(4.303, 3);
    expect(tQuantile975(9)).toBeCloseTo(2.262, 3);
    expect(tQuantile975(30)).toBeCloseTo(2.042, 3);
  });
  it('is monotonically decreasing toward the normal quantile', () => {
    for (let df = 1; df < 60; df++) expect(tQuantile975(df + 1)).toBeLessThanOrEqual(tQuantile975(df));
    expect(tQuantile975(1000)).toBeGreaterThan(1.96);
  });
  it('is unbounded with zero degrees of freedom', () => {
    expect(tQuantile975(0)).toBe(Number.POSITIVE_INFINITY);
  });
});

describe('computeStats', () => {
  it('computes the basic summary', () => {
    const s = computeStats([2, 4, 4, 4, 5, 5, 7, 9]);
    expect(s.mean).toBe(5);
    expect(s.min).toBe(2);
    expect(s.max).toBe(9);
    expect(s.median).toBe(4.5);
    // Sample (n-1) standard deviation.
    expect(s.stddev).toBeCloseTo(Math.sqrt(32 / 7), 10);
    expect(s.ci95).toBeCloseTo((tQuantile975(7) * s.stddev) / Math.sqrt(8), 10);
  });
  it('reports an unbounded CI for a single sample', () => {
    const s = computeStats([3]);
    expect(s.stddev).toBe(0);
    expect(s.ci95).toBe(Number.POSITIVE_INFINITY);
  });
  it('rejects empty input', () => {
    expect(() => computeStats([])).toThrow();
  });
});

describe('hasConverged', () => {
  const opts = { minRuns: 3, precision: 0.03 };

  it('never converges below minRuns, even with identical samples', () => {
    expect(hasConverged([1, 1], opts)).toBe(false);
    expect(hasConverged([1, 1, 1], opts)).toBe(true);
  });

  it('converges on three tight samples', () => {
    // CV ≈ 0.5%; at n=3 the CI half-width is ~2.48 * CV ≈ 1.2% < 3%.
    expect(hasConverged([100, 100.5, 99.5], opts)).toBe(true);
  });

  it('does not converge on three noisy samples but does once more are added', () => {
    // CV ≈ 2%: 3 samples give a ±5% interval, but 8 well-behaved samples get under ±3%.
    const noisy = [98, 100, 102];
    expect(hasConverged(noisy, opts)).toBe(false);
    expect(hasConverged([...noisy, 100, 99, 101, 100, 100], opts)).toBe(true);
  });

  it('is a straight threshold on the relative CI half-width', () => {
    const values = [10, 11, 12, 10, 11];
    const { mean, ci95 } = computeStats(values);
    const rel = ci95 / mean;
    expect(hasConverged(values, { minRuns: 3, precision: rel + 1e-9 })).toBe(true);
    expect(hasConverged(values, { minRuns: 3, precision: rel - 1e-9 })).toBe(false);
  });

  it('never converges on degenerate timings (zero or NaN means)', () => {
    expect(hasConverged([0, 0, 0, 0], opts)).toBe(false);
    expect(hasConverged([NaN, NaN, NaN], opts)).toBe(false);
    expect(hasConverged([-1, -1, -1], opts)).toBe(false);
  });

  it('stops within 10 runs for typical GPU jitter and bails at the cap for wild variance', () => {
    const converging = [5.0, 5.05, 4.98, 5.1, 5.02, 4.97, 5.03, 5.0, 5.04, 4.99];
    let stoppedAt = 0;
    const taken: number[] = [];
    for (const v of converging) {
      taken.push(v);
      if (hasConverged(taken, opts)) {
        stoppedAt = taken.length;
        break;
      }
    }
    expect(stoppedAt).toBeGreaterThanOrEqual(3);
    expect(stoppedAt).toBeLessThan(10);

    const wild = [1, 5, 2, 9, 3, 7, 1, 8, 2, 6];
    for (let n = 1; n <= wild.length; n++) expect(hasConverged(wild.slice(0, n), opts)).toBe(false);
  });
});
