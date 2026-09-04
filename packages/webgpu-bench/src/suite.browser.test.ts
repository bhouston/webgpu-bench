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
