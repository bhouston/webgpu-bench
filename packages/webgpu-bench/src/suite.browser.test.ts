import { test, expect } from 'vitest';
import { runSuite } from './suite.ts';

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
});
