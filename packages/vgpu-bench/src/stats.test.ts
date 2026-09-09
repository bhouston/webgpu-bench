import { describe, expect, it } from 'vitest';
import { computeStats, isBestStable, isThrottled, tQuantile975 } from './stats.ts';

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

describe('isBestStable', () => {
  const opts = { minRuns: 3, stableRuns: 2, tolerance: 0.01 };

  it('never settles below minRuns / stableRuns + 1, even with identical samples', () => {
    expect(isBestStable([1], opts)).toBe(false);
    expect(isBestStable([1, 1], opts)).toBe(false);
    expect(isBestStable([1, 1, 1], opts)).toBe(true);
  });

  it('settles on three samples where the first is the best (typical cool GPU)', () => {
    expect(isBestStable([100, 100.5, 101], opts)).toBe(true);
  });

  it('keeps going while the best is still improving (GPU clock ramping up)', () => {
    expect(isBestStable([110, 105, 100], opts)).toBe(false);
    expect(isBestStable([110, 105, 100, 99], opts)).toBe(false);
    // Two samples that don't beat 99 by more than 1% settle it.
    expect(isBestStable([110, 105, 100, 99, 98.5, 99.2], opts)).toBe(true);
  });

  it('treats a sub-tolerance improvement as noise, not progress', () => {
    expect(isBestStable([100, 99.5, 99.8], opts)).toBe(true);
    expect(isBestStable([100, 98.5, 99.8], opts)).toBe(false);
  });

  it('only looks at the trailing stableRuns samples', () => {
    expect(isBestStable([100, 50, 50, 50], { ...opts, stableRuns: 2 })).toBe(true);
    expect(isBestStable([100, 50, 50, 50], { ...opts, stableRuns: 3 })).toBe(false);
  });

  it('never settles on degenerate timings (zero or NaN)', () => {
    expect(isBestStable([0, 0, 0, 0], opts)).toBe(false);
    expect(isBestStable([NaN, NaN, NaN], opts)).toBe(false);
    expect(isBestStable([-1, -1, -1], opts)).toBe(false);
  });
});

describe('isThrottled', () => {
  it('flags samples more than the threshold slower than the best', () => {
    expect(isThrottled(111, 100, 0.1)).toBe(true);
    expect(isThrottled(110, 100, 0.1)).toBe(false);
    expect(isThrottled(90, 100, 0.1)).toBe(false);
  });
  it('never flags anything before there is a best', () => {
    expect(isThrottled(500, Number.POSITIVE_INFINITY, 0.1)).toBe(false);
    expect(isThrottled(500, 0, 0.1)).toBe(false);
  });
});
