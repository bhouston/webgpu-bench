import { describe, expect, test } from 'vitest';
import { SuiteProgress } from './progress.ts';
import { DEFAULT_SAMPLING } from './sampling.ts';
import type { BenchmarkResult, SuiteProgressEvent } from './types.ts';

const result = (id: string, status: BenchmarkResult['status']) => ({ id, status }) as BenchmarkResult;
const sample = (id: string, count: number, done = false): SuiteProgressEvent => ({
  type: 'sample',
  id,
  durationMs: 100,
  timesMs: Array.from({ length: count }, () => 10),
  throttledMs: [],
  done,
});
const round = (roundNumber: number, ids: string[], overrides = {}): SuiteProgressEvent => ({
  type: 'round',
  round: roundNumber,
  active: ids.length,
  activeIds: ids,
  estimatedRemainingUnits: ids.length * 3,
  sampling: { ...DEFAULT_SAMPLING, ...overrides },
});

describe('coverage-first SuiteProgress', () => {
  test('shows a qualified budget estimate immediately, then completes only on finish', () => {
    const p = new SuiteProgress(10, () => 0);
    expect(p.displayFraction).toBe(0);
    expect(p.remainingSeconds).toBeCloseTo(9);
    expect(p.isApproximate).toBe(true);
    expect(p.remainingSecondsRange![0]).toBeLessThan(p.remainingSeconds!);
    expect(p.remainingSecondsRange![1]).toBeGreaterThan(p.remainingSeconds!);
    p.finish();
    expect(p.displayFraction).toBe(1);
    expect(p.remainingSeconds).toBeNull();
  });

  test('learns whole-benchmark duration without charging preparation to every future benchmark', () => {
    let t = 0;
    const p = new SuiteProgress(5, () => t);
    t = 1000; // upfront preparation
    p.onProgress({ type: 'benchmark-start', id: 'a' });
    p.onProgress(round(1, ['a']));
    t = 1800;
    p.onProgress(sample('a', 3, true));
    p.onResult(result('a', 'ok'));
    p.onResult(result('a', 'ok'));
    expect(p.completedBenchmarks).toBe(1);
    expect(p.remainingSeconds).toBeCloseTo(3.2);
    p.onProgress({ type: 'benchmark-start', id: 'b' });
    t = 2200;
    expect(p.remainingSeconds).toBeCloseTo(2.8);
    expect(p.displayFraction).toBeCloseTo(2.2 / 5);
  });

  test('honors custom sampling and harness budgets before durations are available', () => {
    const p = new SuiteProgress(3, () => 0);
    p.onProgress({
      ...round(1, ['a'], { minRounds: 2, maxRounds: 2, idleMs: 10 }),
      sampleBudgetMs: 50,
      calibrationBudgetMs: 100,
    } as SuiteProgressEvent);
    expect(p.remainingSeconds).toBeCloseTo(0.66);
  });

  test('keeps a countdown during cooldowns and does not train future durations on the cooling pause', () => {
    let t = 0;
    const p = new SuiteProgress(3, () => t);
    p.onProgress({ type: 'benchmark-start', id: 'a' });
    p.onProgress(round(1, ['a'], { maxCooldowns: 1 }));
    t = 500;
    p.onProgress({ type: 'cooldown', attempt: 1, maxAttempts: 1, ms: 3000, throttledIds: ['a'] });
    const eta = p.remainingSeconds!;
    expect(eta).toBeGreaterThan(3);
    expect(p.isApproximate).toBe(true);
    t += 1000;
    expect(p.remainingSeconds).toBeCloseTo(eta - 1);
    t = 3900;
    p.onProgress(sample('a', 3, true));
    expect(p.remainingSeconds).toBeCloseTo(1.8); // 900ms useful duration, two future tests
  });

  test('an overrun does not consume the budgets of later benchmarks or stick at zero', () => {
    let t = 0;
    const p = new SuiteProgress(4, () => t);
    for (const id of ['a', 'b']) {
      p.onProgress({ type: 'benchmark-start', id });
      p.onProgress(round(1, [id]));
      t += 1000;
      p.onProgress(sample(id, 3, true));
    }
    p.onProgress({ type: 'benchmark-start', id: 'c' });
    t += 10_000;
    expect(p.remainingSeconds).toBeGreaterThan(1);
    expect(p.displayFraction).toBeLessThan(1);
    expect(p.isApproximate).toBe(true);
  });

  test('unbounded visibility pauses hide time estimates and resume on activity', () => {
    const p = new SuiteProgress(2, () => 0);
    p.onProgress({ type: 'pause' });
    expect(p.displayFraction).toBeNull();
    expect(p.remainingSeconds).toBeNull();
    p.onProgress({ type: 'benchmark-start', id: 'a' });
    expect(p.remainingSeconds).not.toBeNull();
  });

  test('skipped and failed setup rows remove work without training a zero-duration benchmark', () => {
    const p = new SuiteProgress(3, () => 0);
    p.onResult(result('a', 'skipped'));
    p.onResult(result('b', 'error'));
    expect(p.remainingSeconds).toBeCloseTo(0.9);
    expect(p.completedBenchmarks).toBe(2);
    expect(p.benchmarkCount).toBe(3);
    p.onResult(result('c', 'ok'));
    expect(p.remainingSeconds).toBe(0);
    expect(p.displayFraction).toBeLessThan(1);
    p.finish();
    expect(p.displayFraction).toBe(1);
  });

  test('empty suites have no fabricated estimate', () => {
    const p = new SuiteProgress(0);
    expect(p.displayFraction).toBeNull();
    expect(p.remainingSeconds).toBeNull();
    p.finish();
    expect(p.fraction).toBe(1);
  });

  test('round-robin also estimates unseen calibration and work instead of waiting for two full rounds', () => {
    let t = 0;
    const p = new SuiteProgress(10, () => t);
    p.onProgress(
      round(
        1,
        Array.from({ length: 10 }, (_, i) => `k${i}`),
        { samplingOrder: 'round-robin' },
      ),
    );
    t = 400;
    p.onProgress(sample('k0', 1));
    expect(p.remainingSeconds).toBeGreaterThan(5);
    expect(p.displayFraction).not.toBeNull();
    expect(p.isApproximate).toBe(true);
  });

  test.each([0, NaN, Infinity])('invalid duration %s cannot turn an estimate into NaN', (durationMs) => {
    const p = new SuiteProgress(3, () => 100);
    p.onProgress(round(1, ['a', 'b', 'c'], { samplingOrder: 'round-robin' }));
    p.onProgress({ ...sample('a', 1), durationMs } as SuiteProgressEvent);
    expect(Number.isFinite(p.remainingSeconds)).toBe(true);
    expect(Number.isFinite(p.displayFraction)).toBe(true);
  });
});
