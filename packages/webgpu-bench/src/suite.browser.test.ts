import { test, expect } from 'vitest';
import { runSuite } from './suite.ts';
import { BENCHMARK_CATALOG } from './catalog.ts';

test('every kernel runs clean', { timeout: 60_000 }, async () => {
  const results = [];
  for await (const r of runSuite({ computeThreads: 4096, targetMs: 200 })) {
    if (r.status === 'running') continue;
    results.push(r);
  }
  const byId = new Map(results.map((r) => [r.id, r]));
  for (const [id, r] of byId) {
    expect(r.status, `${id}: ${r.message ?? ''}`).not.toBe('error');
    if (r.status === 'ok') {
      expect(Number.isFinite(r.metricValue), `${id} ${r.category}`).toBe(true);
      expect(r.metricValue, `${id} ${r.category}`).toBeGreaterThan(0);
    }
  }
  // BENCHMARK_CATALOG is a hand-written mirror of runSuite's kernel list (kept
  // GPU-free so a UI can render placeholder rows before a device is even
  // acquired) — this is the tripwire that catches it drifting from the real thing.
  expect(new Set(BENCHMARK_CATALOG.map((b) => b.id))).toEqual(new Set(byId.keys()));
});

test(
  'the page stays responsive during a real run: a 200ms poll timer never falls badly behind',
  { timeout: 30_000 },
  async () => {
    // Responsiveness is a property of runSampling's round-robin/idle-gap
    // scheduling, not of any one kernel — one round across every real kernel
    // exercises that strategy without paying for full convergence.
    const pollMs = 200;
    const ticks: number[] = [];
    const start = performance.now();
    const timer = setInterval(() => ticks.push(performance.now() - start), pollMs);
    try {
      for await (const r of runSuite({ computeThreads: 4096, targetMs: 50, minRounds: 1, maxRounds: 1, warmups: 0 })) {
        void r;
      }
    } finally {
      clearInterval(timer);
    }

    expect(ticks.length).toBeGreaterThan(5);
    for (let i = 1; i < ticks.length; i++) {
      // Generous tolerance for scheduler jitter; a real main-thread stall
      // (e.g. a synchronous GPU readback) would blow way past this.
      expect(ticks[i]! - ticks[i - 1]!).toBeLessThan(pollMs + 150);
    }
  },
);
