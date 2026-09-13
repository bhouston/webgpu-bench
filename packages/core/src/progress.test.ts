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
      sampling: { ...DEFAULT_SAMPLING, minRounds: 6, maxRounds: 6 },
      estimatedRemainingUnits: 10,
    });
    time = 1000;
    p.onProgress(sample('a', 100));
    expect(p.displayFraction).toBeNull();
    time = 2000;
    p.onProgress(sample('b', 100));
    p.onResult(row);
    expect(p.completedUnits).toBe(2);
    expect(p.displayFraction).toBeNull(); // need repeated timing evidence
    p.onProgress({
      type: 'round',
      round: 2,
      active: 2,
      activeIds: ['a', 'b'],
      sampling: { ...DEFAULT_SAMPLING, minRounds: 6, maxRounds: 6 },
      estimatedRemainingUnits: 10,
    });
    time = 2100;
    p.onProgress(sample('a', 100, 2));
    time = 2300;
    p.onProgress(sample('b', 100, 2));
    // Calibration contributes to elapsed time, never to future sample cost.
    expect(p.remainingSeconds).toBeCloseTo(1.2);
    expect(p.displayFraction).toBeCloseTo(2.3 / 3.5);
    // A numeric percentage and ETA can be withdrawn after they have been shown.
    p.onProgress({ type: 'cooldown', attempt: 1, maxAttempts: 3, ms: 3000, throttledIds: ['a', 'b'] });
    expect(p.displayFraction).toBeNull();
    expect(p.remainingSeconds).toBeNull();
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

function fixedRun() {
  let time = 0;
  const p = new SuiteProgress(1, () => time);
  const round = (n: number) =>
    p.onProgress({
      type: 'round',
      round: n,
      active: 1,
      activeIds: ['a'],
      estimatedRemainingUnits: 7 - n,
      sampling: { ...DEFAULT_SAMPLING, minRounds: 6, maxRounds: 6, idleMs: 0 },
    });
  round(1);
  time = 1000;
  p.onProgress(sample('a', 100));
  round(2);
  time = 1100;
  p.onProgress(sample('a', 100, 2));
  return {
    p,
    round,
    setTime: (t: number) => {
      time = t;
    },
  };
}

test('expires a previously visible estimate while a dispatch is stalled', () => {
  const { p, round, setTime } = fixedRun();
  expect(p.displayFraction).not.toBeNull();
  expect(p.remainingSeconds).toBeCloseTo(0.4);
  round(3);
  setTime(1400);
  expect(p.displayFraction).toBeNull();
  expect(p.remainingSeconds).toBeNull();
});

test.each(['pause', 'throttle-abort'] as const)('withdraws estimates on %s', (type) => {
  const { p } = fixedRun();
  expect(p.displayFraction).not.toBeNull();
  p.onProgress(type === 'pause' ? { type } : { type, throttledIds: ['a'] });
  expect(p.displayFraction).toBeNull();
  expect(p.remainingSeconds).toBeNull();
});

test('withdraws on a discarded sample and can recover with subsequent evidence', () => {
  const { p, round, setTime } = fixedRun();
  round(3);
  setTime(1200);
  p.onProgress({ type: 'sample', id: 'a', durationMs: 100, timesMs: [10, 10], throttledMs: [15], done: false });
  expect(p.displayFraction).toBeNull();
  round(4);
  setTime(1300);
  p.onProgress({ type: 'sample', id: 'a', durationMs: 100, timesMs: [10, 10, 10], throttledMs: [15], done: false });
  expect(p.displayFraction).toBeNull();
  round(5);
  setTime(1400);
  p.onProgress({ type: 'sample', id: 'a', durationMs: 100, timesMs: [10, 10, 10, 10], throttledMs: [15], done: false });
  expect(p.displayFraction).not.toBeNull();
});

test.each([0, Number.NaN, Infinity, 300])('rejects missing or surprising duration %s', (duration) => {
  const { p, round, setTime } = fixedRun();
  round(3);
  setTime(1400);
  p.onProgress(sample('a', duration, 3));
  expect(p.displayFraction).toBeNull();
  expect(p.remainingSeconds).toBeNull();
});

test('can show a useful late percentage while withholding an uncertain ETA', () => {
  let time = 0;
  const p = new SuiteProgress(1, () => time);
  for (let n = 1; n <= 2; n++) {
    p.onProgress({
      type: 'round',
      round: n,
      active: 1,
      activeIds: ['a'],
      sampling: DEFAULT_SAMPLING,
      estimatedRemainingUnits: 6 - n,
    });
    time += 1000;
    p.onProgress(sample('a', 100, n));
  }
  expect(p.displayFraction).toBeGreaterThan(0.9);
  expect(p.remainingSeconds).toBeNull();
  expect(p.displayFraction).toBeLessThan(1);
});
