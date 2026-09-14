import { describe, expect, it } from 'vitest';
import { recordSample, resolveSamplingConfig, runSampling, type Sampleable, type SampleState } from './sampling.ts';
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

describe('runSampling', () => {
  it('retires converged benchmarks while others keep sampling', async () => {
    const log: string[] = [];
    const quick = scripted('quick', [10, 10, 10], log);
    const ramping = scripted('ramping', [30, 27, 24, 21, 20, 20, 20], log);
    const results = await runSampling([quick, ramping], { idleMs: 0 });
    expect(log.filter((l) => l === 'quick:sample')).toHaveLength(3);
    expect(results.get('ramping')!.stats!.min).toBe(20);
    expect(results.get('ramping')!.stopReason).toBe('converged');
  });

  it('pauses after consecutive discards, then recovers and keeps the best', async () => {
    const events: SuiteProgressEvent[] = [];
    const sleeps: number[] = [];
    // Both benchmarks: one clean sample, then two discarded samples (device hot), then clean again.
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

  it('caps a benchmark that only ever comes back throttled on its own', async () => {
    const events: SuiteProgressEvent[] = [];
    const stuck = scripted('stuck', [10, 20]); // one good run, then slow forever
    const results = await runSampling([stuck], {
      idleMs: 0,
      maxRounds: 5,
      maxCooldowns: 10,
      sleep: async () => {},
      onProgress: (e) => events.push(e),
    });
    expect(events.filter((e) => e.type === 'cooldown')).toHaveLength(4);
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

  it('reports remaining sample units for the current benchmark', async () => {
    const events: SuiteProgressEvent[] = [];
    // cfg: minRounds=3, stableRounds=2 (defaults) → converges once the best hasn't improved over 2 kept samples.
    const quick = scripted('quick', [10, 10, 10]);
    const slow = scripted('slow', [20, 18, 16, 16, 16]);
    await runSampling([quick, slow], { ...cfg, onProgress: (e) => events.push(e) });
    const rounds = events.filter((e): e is Extract<SuiteProgressEvent, { type: 'round' }> => e.type === 'round');

    // Unit telemetry describes the current benchmark's remaining samples.
    expect(rounds[0]!.estimatedRemainingUnits).toBe(cfg.minRounds + cfg.stableRounds);
    expect(rounds.every((e) => e.active === 1)).toBe(true);
    // The final event reflects the last benchmark, floored at one sample.
    const lastRound = rounds.at(-1)!;
    expect(lastRound.active).toBe(1);
    expect(lastRound.estimatedRemainingUnits).toBeGreaterThanOrEqual(1);
  });

  it('reports the best-so-far after every sample via onUpdate', async () => {
    const bests: number[] = [];
    const a = scripted('a', [12, 11, 10, 10, 10]);
    await runSampling([a], { idleMs: 0, onUpdate: (s) => void bests.push(s.bestMs) });
    expect(bests).toEqual([12, 11, 10, 10, 10]);
  });
});

it('reports wall-time telemetry without charging calibration or idle to samples', async () => {
  let time = 0;
  const events: SuiteProgressEvent[] = [];
  await runSampling(
    ['a', 'b'].map((id) => ({
      id,
      calibrate: async () => {
        time += 900;
      },
      sample: async () => {
        time += 25;
        return 10;
      },
    })),
    {
      minRounds: 1,
      maxRounds: 1,
      idleMs: 100,
      now: () => time,
      sleep: async (ms) => {
        time += ms;
      },
      onProgress: (e) => events.push(e),
    },
  );
  expect(time).toBe(1950);
  expect(events.filter((e) => e.type === 'sample')).toEqual([
    expect.objectContaining({ durationMs: 25, done: true, timesMs: [10] }),
    expect.objectContaining({ durationMs: 25, done: true, timesMs: [10] }),
  ]);
  expect(events[1]).toMatchObject({ type: 'round', sampling: { minRounds: 1, maxRounds: 1, idleMs: 100 } });
});

it('completes benchmarks in input order by default, calibrating each once and resting between samples', async () => {
  const log: string[] = [],
    sleeps: number[] = [];
  const result = await runSampling([scripted('a', [10], log), scripted('b', [20], log)], {
    idleMs: 100,
    sleep: async (ms) => {
      sleeps.push(ms);
    },
  });
  expect(log).toEqual([
    'a:calibrate',
    'a:sample',
    'a:sample',
    'a:sample',
    'b:calibrate',
    'b:sample',
    'b:sample',
    'b:sample',
  ]);
  expect(sleeps).toEqual([100, 100, 100, 100, 100]);
  expect([...result.values()].every((s) => s.stopReason === 'converged')).toBe(true);
});

it('cools a sequential kernel after two discards and continues with untouched kernels after its budget expires', async () => {
  const events: SuiteProgressEvent[] = [],
    sleeps: number[] = [];
  const result = await runSampling([scripted('hot', [10, 15]), scripted('good', [10])], {
    idleMs: 0,
    cooldownMs: 3000,
    maxCooldowns: 1,
    sleep: async (ms) => {
      sleeps.push(ms);
    },
    onProgress: (e) => events.push(e),
  });
  expect(sleeps).toEqual([3000]);
  expect(result.get('hot')!.stopReason).toBe('throttled');
  expect(result.get('good')!.stopReason).toBe('converged');
  expect(result.get('good')!.timesMs).toEqual([10, 10, 10]);
  expect(events.filter((e) => e.type === 'benchmark-start')).toEqual([
    { type: 'benchmark-start', id: 'hot' },
    { type: 'benchmark-start', id: 'good' },
  ]);
});

it('shares a finite cooldown budget across sequential benchmarks', async () => {
  const pauses: number[] = [];
  const result = await runSampling([scripted('hot-a', [10, 15]), scripted('hot-b', [10, 15]), scripted('good', [10])], {
    maxCooldowns: 1,
    idleMs: 0,
    cooldownMs: 3000,
    sleep: async (ms) => {
      pauses.push(ms);
    },
  });
  expect(pauses).toEqual([3000]);
  expect(result.get('hot-a')!.stopReason).toBe('throttled');
  expect(result.get('hot-b')!.stopReason).toBe('throttled');
  expect(result.get('good')!.stopReason).toBe('converged');
});
