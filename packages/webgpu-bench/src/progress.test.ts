import { describe, expect, test } from 'vitest';
import { SuiteProgress } from './progress.ts';
import type { BenchmarkResult } from './types.ts';

const row = { id: 'x' } as BenchmarkResult;

describe('SuiteProgress', () => {
  test('fraction climbs with rows, tracks the library estimate, and finishes at 1', () => {
    let t = 0;
    const p = new SuiteProgress(2, () => t);
    expect(p.fraction).toBe(0);
    p.onResult(row);
    p.onResult(row); // setup rows for both benchmarks
    p.onProgress({ type: 'round', round: 1, active: 2, estimatedRemainingUnits: 8 });
    expect(p.fraction).toBeCloseTo(2 / 10);
    expect(p.remainingSeconds).toBeNull();
    for (let i = 0; i < 4; i++) {
      t += 1000;
      p.onResult(row);
    }
    p.onProgress({ type: 'round', round: 3, active: 2, estimatedRemainingUnits: 4 });
    expect(p.fraction).toBeCloseTo(6 / 10);
    expect(p.remainingSeconds).toBe(4);
    p.finish();
    expect(p.fraction).toBe(1);
    expect(p.remainingSeconds).toBeNull();
  });
});
