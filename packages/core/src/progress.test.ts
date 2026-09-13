import { describe, expect, test } from 'vitest';
import { SuiteProgress } from './progress.ts';
import { DEFAULT_SAMPLING } from './sampling.ts';
import type { BenchmarkResult, SuiteProgressEvent } from './types.ts';

const row = { id: 'x', status: 'running' } as BenchmarkResult;
const sample = (id: string, durationMs: number, count = 1, done = false): SuiteProgressEvent => ({
  type: 'sample',
  id,
  durationMs,
  timesMs: Array.from({ length: count }, () => 10),
  throttledMs: [],
  done,
});

describe('SuiteProgress', () => {
  test('hides setup and incomplete calibration, and counts samples exactly once', () => {
    let time = 0;
    const p = new SuiteProgress(2, () => time);
    p.onResult(row);
    p.onResult(row);
    expect(p.displayFraction).toBeNull();
    expect(p.remainingSeconds).toBeNull();
    expect(p.completedUnits).toBe(0);
    p.onProgress({
      type: 'round',
      round: 1,
      active: 2,
      activeIds: ['a', 'b'],
      sampling: DEFAULT_SAMPLING,
      estimatedRemainingUnits: 10,
    });
    time = 1000;
    p.onProgress(sample('a', 100));
    expect(p.displayFraction).toBeNull();
    time = 2000;
    p.onProgress(sample('b', 100));
    p.onResult(row);
    expect(p.completedUnits).toBe(2);
    // Calibration contributes to elapsed time, never to the cost of future samples.
    expect(p.remainingSeconds).toBeGreaterThan(0.6);
    expect(p.remainingSeconds).toBeLessThan(0.7);
    expect(p.displayFraction).toBeGreaterThan(0.74);
    expect(p.displayFraction).toBeLessThan(0.77);
    p.finish();
    expect(p.fraction).toBe(1);
    expect(p.displayFraction).toBe(1);
    expect(p.remainingSeconds).toBeNull();
  });

  test('empty and all-skipped suites become complete only at finish', () => {
    const p = new SuiteProgress(0);
    expect(p.displayFraction).toBeNull();
    p.finish();
    expect(p.displayFraction).toBe(1);
  });

  test('legacy event streams remain indeterminate without duration evidence', () => {
    const p = new SuiteProgress(1);
    p.onProgress({ type: 'round', round: 1, active: 1, estimatedRemainingUnits: 5 });
    for (let i = 0; i < 10; i++) p.onResult(row);
    expect(p.displayFraction).toBeNull();
    expect(p.remainingSeconds).toBeNull();
  });
});
