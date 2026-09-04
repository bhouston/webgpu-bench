import { describe, expect, it } from 'vitest';
import {
  recordSample,
  resolveSamplingConfig,
  roundIsThrottled,
  runSampling,
  type Sampleable,
  type SampleState,
} from './sampling.ts';
import type { SuiteProgressEvent } from './types.ts';

const cfg = resolveSamplingConfig({ idleMs: 0, cooldownMs: 0, throttleThreshold: 0.1 });

function freshState(id = 'k'): SampleState {
  return { id, timesMs: [], throttledMs: [], bestMs: Number.POSITIVE_INFINITY };
}

/** A fake kernel that hands out the given timings in order (last one repeats). */
function scripted(id: string, timings: number[], log: string[] = []): Sampleable {
  let i = 0;
  return {
    id,
    calibrate: async () => {
      log.push(`${id}:calibrate`);
    },
    sample: async () => {
      log.push(`${id}:sample`);
      const v = timings[Math.min(i, timings.length - 1)]!;
      i++;
      return v;
    },
  };
}

describe('recordSample', () => {
  it('keeps samples, tracks the best, and converges once the best stops improving', () => {
    const s = freshState();
    expect(recordSample(s, 10, cfg)).toBe(true);
    expect(recordSample(s, 10.05, cfg)).toBe(true);
    expect(s.stopReason).toBeUndefined();
    expect(recordSample(s, 9.98, cfg)).toBe(true);
    expect(s.bestMs).toBe(9.98);
    expect(s.stopReason).toBe('converged');
  });

  it('discards throttled samples without touching the best or convergence', () => {
    const s = freshState();
    recordSample(s, 10, cfg);
    expect(recordSample(s, 12, cfg)).toBe(false);
    expect(s.timesMs).toEqual([10]);
    expect(s.throttledMs).toEqual([12]);
    expect(s.bestMs).toBe(10);
  });

  it('stops at maxRounds when the best keeps improving', () => {
    const s = freshState();
    const c = resolveSamplingConfig({ maxRounds: 4 });
    for (const v of [100, 95, 90, 85]) recordSample(s, v, c);
    expect(s.stopReason).toBe('max-rounds');
    expect(s.timesMs).toHaveLength(4);
  });
});

describe('roundIsThrottled', () => {
  it('needs at least two throttled benchmarks and the configured fraction', () => {
    expect(roundIsThrottled([true, false, false, false], cfg)).toBe(false);
    expect(roundIsThrottled([true, true, false, false], cfg)).toBe(true);
    expect(roundIsThrottled([true, true, false], cfg)).toBe(true);
    expect(roundIsThrottled([true, false, false], cfg)).toBe(false);
  });

  it('ignores one persistently noisy benchmark among healthy ones', () => {
    expect(roundIsThrottled([true, false, false, false, false, false], cfg)).toBe(false);
  });

  it('never triggers on a single benchmark', () => {
    expect(roundIsThrottled([true], cfg)).toBe(false);
  });
});

describe('runSampling', () => {
  it('interleaves benchmarks round-robin and calibrates each exactly once', async () => {
    const log: string[] = [];
    const a = scripted('a', [10, 10, 10], log);
    const b = scripted('b', [20, 20, 20], log);
    const results = await runSampling([a, b], { idleMs: 0 });
    expect(log.filter((e) => e.endsWith(':calibrate'))).toHaveLength(2);
    // Every round measures both once; the order within a round is random.
    const samples = log.filter((e) => e.endsWith(':sample'));
    expect(samples).toHaveLength(6);
    for (let i = 0; i < samples.length; i += 2) {
      expect(new Set(samples.slice(i, i + 2))).toEqual(new Set(['a:sample', 'b:sample']));
    }
    expect(results.get('a')!.stopReason).toBe('converged');
    expect(results.get('a')!.stats!.min).toBe(10);
    expect(results.get('b')!.timesMs).toEqual([20, 20, 20]);
  });

  it('retires converged benchmarks while others keep sampling', async () => {
    const log: string[] = [];
    const quick = scripted('quick', [10, 10, 10], log);
    const ramping = scripted('ramping', [30, 27, 24, 21, 20, 20, 20], log);
    const results = await runSampling([quick, ramping], { idleMs: 0 });
    expect(log.filter((l) => l === 'quick:sample')).toHaveLength(3);
    expect(results.get('ramping')!.stats!.min).toBe(20);
    expect(results.get('ramping')!.stopReason).toBe('converged');
  });

  it('pauses the suite on a throttled round, then recovers and keeps the best', async () => {
    const events: SuiteProgressEvent[] = [];
    const sleeps: number[] = [];
    // Both benchmarks: one clean sample, then two throttled rounds (device hot), then clean again.
    const a = scripted('a', [10, 15, 15, 10, 10]);
    const b = scripted('b', [20, 30, 30, 20, 20]);
    const results = await runSampling([a, b], {
      idleMs: 0,
      cooldownMs: 1234,
      onProgress: (e) => events.push(e),
      sleep: async (ms) => {
        sleeps.push(ms);
      },
    });
    const cooldowns = events.filter((e) => e.type === 'cooldown');
    expect(cooldowns.length).toBeGreaterThanOrEqual(1);
    expect(sleeps).toContain(1234);
    expect(results.get('a')!.stats!.min).toBe(10);
    expect(results.get('a')!.throttledMs).toEqual([15, 15]);
    expect(results.get('a')!.timesMs).toEqual([10, 10, 10]);
    expect(results.get('a')!.stopReason).toBe('converged');
    expect(results.get('b')!.stopReason).toBe('converged');
  });

  it('does not pause for one noisy benchmark while the others are healthy', async () => {
    const events: SuiteProgressEvent[] = [];
    const noisy = scripted('noisy', [10, 13, 13, 10, 12, 10]);
    const a = scripted('a', [20, 20, 20]);
    const b = scripted('b', [30, 30, 30]);
    const results = await runSampling([noisy, a, b], {
      idleMs: 0,
      throttleThreshold: 0.1,
      onProgress: (e) => events.push(e),
      sleep: async () => {},
    });
    expect(events.filter((e) => e.type === 'cooldown')).toHaveLength(0);
    expect(results.get('noisy')!.stopReason).toBe('converged');
    expect(results.get('noisy')!.throttledMs).toEqual([13, 13, 12]);
  });

  it('caps a benchmark that only ever comes back throttled on its own', async () => {
    const events: SuiteProgressEvent[] = [];
    const stuck = scripted('stuck', [10, 20]); // one good run, then slow forever
    const results = await runSampling([stuck], { idleMs: 0, maxRounds: 5, onProgress: (e) => events.push(e) });
    expect(events.filter((e) => e.type === 'cooldown')).toHaveLength(0);
    expect(results.get('stuck')!.stopReason).toBe('max-rounds');
    expect(results.get('stuck')!.timesMs).toEqual([10]);
    expect(results.get('stuck')!.throttledMs).toHaveLength(9);
  });

  it('gives up with stopReason "throttled" once the cooldown budget is spent', async () => {
    const events: SuiteProgressEvent[] = [];
    const a = scripted('a', [10, 15]); // never recovers
    const b = scripted('b', [20, 30]);
    const results = await runSampling([a, b], {
      idleMs: 0,
      maxCooldowns: 2,
      onProgress: (e) => events.push(e),
      sleep: async () => {},
    });
    expect(events.filter((e) => e.type === 'cooldown')).toHaveLength(2);
    expect(events.at(-1)!.type).toBe('throttle-abort');
    expect(results.get('a')!.stopReason).toBe('throttled');
    expect(results.get('a')!.stats!.min).toBe(10);
    expect(results.get('b')!.stopReason).toBe('throttled');
  });

  it('retires a benchmark whose sampler throws, without stopping the others', async () => {
    const bad: Sampleable = {
      id: 'bad',
      calibrate: async () => {},
      sample: async () => {
        throw new Error('boom');
      },
    };
    const good = scripted('good', [5, 5, 5]);
    const updates: string[] = [];
    const results = await runSampling([bad, good], { idleMs: 0, onUpdate: (s) => void updates.push(s.id) });
    expect((results.get('bad')!.error as Error).message).toBe('boom');
    expect(results.get('good')!.stopReason).toBe('converged');
    expect(updates.filter((u) => u === 'bad')).toHaveLength(1);
  });

  it('retires a benchmark that measures zero time as an error', async () => {
    const zero = scripted('zero', [0]);
    const results = await runSampling([zero], { idleMs: 0 });
    expect(results.get('zero')!.error).toBeInstanceOf(Error);
    expect(String((results.get('zero')!.error as Error).message)).toMatch(/did no work/);
  });

  it('reports the best-so-far after every sample via onUpdate', async () => {
    const bests: number[] = [];
    const a = scripted('a', [12, 11, 10, 10, 10]);
    await runSampling([a], { idleMs: 0, onUpdate: (s) => void bests.push(s.bestMs) });
    expect(bests).toEqual([12, 11, 10, 10, 10]);
  });
});
