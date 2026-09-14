// Opt-in, deterministic replay. All predictions see only the trace prefix.
// GPU collection is sequential and never submits benchmark results anywhere.
import { createServer } from 'node:http';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve, sep } from 'node:path';
import { parseArgs } from 'node:util';
import { gunzipSync } from 'node:zlib';
import { runSampling } from '../packages/core/dist/sampling.js';
import { SuiteProgress } from '../packages/core/dist/progress.js';

const { values } = parseArgs({
  options: {
    capture: { type: 'string' },
    input: { type: 'string' },
    output: { type: 'string' },
    traces: { type: 'string' },
    repeats: { type: 'string', default: '3' },
    seed: { type: 'string', default: '1' },
    groups: { type: 'string', default: 'representative,single,fixed' },
  },
});

function random(seed) {
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
}

async function synthetic() {
  const traces = [];
  for (const family of [
    'stable',
    'ramping',
    'mixed',
    'noisy-tail',
    'cooldown',
    'late-slowdown',
    'expensive-tail',
    'errors',
    'fixed-rounds',
    'long-calibration',
    'tiny',
    'late-improvement',
  ]) {
    for (let seed = 1; seed <= 12; seed++) {
      const rng = random((seed + Number(values.seed) - 1) * 71);
      let time = 0;
      const events = [];
      const count = family === 'tiny' ? 1 : 4 + (seed % 9);
      const push = (kind, value) => events.push({ at: time, kind, value });
      const rows = Array.from({ length: count }, (_, i) => ({
        id: `k${i}`,
        status: 'running',
        timesMs: [],
        throttledMs: [],
      }));
      for (const row of rows) {
        time += 10 + rng() * 30;
        push('result', row);
      }
      const kernels = rows.map((row, index) => {
        let attempt = 0;
        const cost = family === 'expensive-tail' && index === 0 ? 1500 : 50 + rng() * 200;
        return {
          id: row.id,
          calibrate: async () => {
            time += (family === 'long-calibration' ? 2000 : 150) + rng() * 150;
          },
          sample: async () => {
            attempt++;
            time += cost * (0.96 + rng() * 0.08) * (family === 'late-slowdown' && attempt >= 3 ? 3 : 1);
            if (family === 'errors' && index % 3 === 0 && attempt === 2) throw new Error('scripted error');
            if (family === 'cooldown' && attempt >= 2 && attempt <= 4) return 18;
            if (family === 'noisy-tail' && index === 0 && attempt > 1 && attempt < 10) return 18;
            if (family === 'late-improvement' && attempt === 3) return 8;
            if (family === 'late-improvement' && attempt > 3) return 8;
            if (family === 'ramping' || (['mixed', 'expensive-tail'].includes(family) && index % 3 === 0))
              return 10 * 0.92 ** Math.min(attempt, 4 + (seed % 5));
            return 10 * (1 + rng() * 0.005);
          },
        };
      });
      await runSampling(kernels, {
        ...(family === 'fixed-rounds' ? { minRounds: 6, maxRounds: 6 } : {}),
        now: () => time,
        sleep: async (ms) => {
          time += ms;
        },
        onProgress: (e) => push('progress', e),
        onUpdate: (s) =>
          push('result', {
            id: s.id,
            status: s.error ? 'error' : s.stopReason ? 'ok' : 'running',
            timesMs: s.timesMs,
            throttledMs: s.throttledMs,
          }),
      });
      time += 5;
      traces.push({ name: `${family}-${seed}`, group: family, count, totalMs: time, events });
    }
  }
  return traces;
}

async function captureBrowser(name) {
  if (!['webkit', 'chromium'].includes(name)) throw new Error('Unknown browser');
  const { chromium, webkit } = await import('playwright');
  const root = resolve('packages/core/dist');
  const server = createServer((req, res) => {
    if (req.url === '/') {
      res.setHeader('Content-Type', 'text/html');
      res.end('<!doctype html><title>Progress study</title>');
      return;
    }
    const path = resolve(root, '.' + new URL(req.url, 'http://localhost').pathname);
    if (!path.startsWith(root + sep) || !path.endsWith('.js')) {
      res.writeHead(404).end();
      return;
    }
    try {
      res.setHeader('Content-Type', 'text/javascript');
      res.end(readFileSync(path));
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise((done) => server.listen(0, '127.0.0.1', done));
  let browser;
  const traces = [];
  try {
    browser = await (name === 'webkit' ? webkit : chromium).launch(
      name === 'webkit'
        ? { headless: true }
        : { headless: false, args: ['--headless=new', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] },
    );
    for (let repeat = 0; repeat < Number(values.repeats); repeat++) {
      for (const group of values.groups.split(',')) {
        const page = await browser.newPage();
        await page.goto(`http://127.0.0.1:${server.address().port}/`);
        const trace = await page.evaluate(
          async ({ group: workload }) => {
            const { runSuite, BENCHMARKS } = await import('/suite.js');
            const ids =
              workload === 'single'
                ? ['f32-div']
                : [
                    'read-linear',
                    'write-scatter-16kb',
                    'texture-interp-manual',
                    'f32-fma-scalar',
                    'f16-fma-mat4',
                    'f32-div',
                    'workgroup-64',
                    'layout-soa',
                    'reduction-workgroup',
                    'atomic-direct',
                    'read-dependent-chain',
                  ];
            const benchmarks = workload === 'all' ? BENCHMARKS : BENCHMARKS.filter((b) => ids.includes(b.id));
            const events = [];
            const start = performance.now();
            for await (const row of runSuite({
              benchmarks,
              ...(workload === 'fixed' ? { minRounds: 6, maxRounds: 6 } : {}),
              onProgress: (e) => events.push({ at: performance.now() - start, kind: 'progress', value: e }),
            })) {
              if (row.status === 'error') throw new Error(`${row.id}: ${row.message}`);
              events.push({
                at: performance.now() - start,
                kind: 'result',
                value: { id: row.id, status: row.status, timesMs: row.timesMs, throttledMs: row.throttledMs },
              });
            }
            return { count: benchmarks.length, totalMs: performance.now() - start, events };
          },
          { group },
        );
        traces.push({ name: `${name}-${group}-${repeat}`, group: `${name}-${group}`, ...trace });
        console.error(`${traces.at(-1).name}: ${(trace.totalMs / 1000).toFixed(2)}s`);
        await page.close();
        if (values.traces) writeFileSync(values.traces, JSON.stringify(traces));
      }
    }
  } finally {
    await browser?.close();
    server.close();
  }
  return traces;
}

const format = (g) =>
  Object.fromEntries(
    ['fraction', 'eta'].map((k) => [
      k,
      {
        coverage: +((100 * g[`${k}Visible`]) / g.runtime).toFixed(2),
        within20: g[`${k}Visible`] ? +((100 * g[`${k}Good`]) / g[`${k}Visible`]).toFixed(2) : null,
        meanRelativeError: g[`${k}Visible`] ? +((100 * g[`${k}Error`]) / g[`${k}Visible`]).toFixed(2) : null,
        accurateCoverage: +((100 * g[`${k}Good`]) / g.runtime).toFixed(2),
      },
    ]),
  );

function evaluate(traces, display = false) {
  const groups = {};
  for (const trace of traces) {
    let now = 0,
      index = 0;
    if (!trace.events.some((e) => e.kind === 'progress' && e.value.type === 'benchmark-start'))
      throw new Error(
        `Trace ${trace.name} lacks sequential benchmark-start telemetry; use its historical revision for replay.`,
      );
    const p = new SuiteProgress(trace.count, () => now);
    let nextRefresh = 100,
      displayedFraction = null,
      displayedEta = null;
    const render = () => {
      const fraction = p.displayFraction;
      const percent = fraction === null ? null : Math.floor(fraction * 100);
      displayedFraction = percent === null ? null : percent / 100;
      const eta = p.remainingSeconds;
      displayedEta = fraction === null || eta === null ? null : +Math.max(0.1, eta).toFixed(1);
    };
    if (display) render();
    const summary = (groups[trace.group] ??= {
      runtime: 0,
      fractionVisible: 0,
      fractionGood: 0,
      fractionError: 0,
      etaVisible: 0,
      etaGood: 0,
      etaError: 0,
    });
    // Midpoint integration on a uniform grid, including time between events.
    const step = Math.min(50, trace.totalMs / 1000);
    for (let at = step / 2; at < trace.totalMs; at += step) {
      while (true) {
        const eventAt = trace.events[index]?.at ?? Infinity;
        const tickAt = display ? nextRefresh : Infinity;
        if (Math.min(eventAt, tickAt) > at) break;
        if (tickAt < eventAt) {
          now = tickAt;
          nextRefresh += 100;
          render();
        } else {
          const e = trace.events[index++];
          now = e.at;
          if (e.kind === 'progress') p.onProgress(e.value);
          else p.onResult(e.value);
          if (display) render();
        }
      }
      now = at;
      summary.runtime += step;
      const fraction = display ? displayedFraction : p.displayFraction;
      const eta = display ? displayedEta : p.remainingSeconds;
      for (const [key, estimate, truth] of [
        ['fraction', fraction, at / trace.totalMs],
        ['eta', eta, (trace.totalMs - at) / 1000],
      ]) {
        if (estimate == null) continue;
        const error = Math.abs(estimate / truth - 1);
        summary[`${key}Visible`] += step;
        summary[`${key}Good`] += error <= 0.2 ? step : 0;
        summary[`${key}Error`] += error * step;
      }
    }
  }
  const total = Object.values(groups).reduce((a, g) => {
    for (const [k, v] of Object.entries(g)) a[k] = (a[k] ?? 0) + v;
    return a;
  }, {});
  return { total: format(total), groups: Object.fromEntries(Object.entries(groups).map(([k, g]) => [k, format(g)])) };
}

const traces = values.input
  ? JSON.parse(
      values.input.endsWith('.gz')
        ? gunzipSync(readFileSync(values.input)).toString('utf8')
        : readFileSync(values.input, 'utf8'),
    )
  : values.capture === 'synthetic'
    ? await synthetic()
    : await captureBrowser(values.capture);
if (values.traces) writeFileSync(values.traces, JSON.stringify(traces));
const report = {
  runs: traces.length,
  candidate: evaluate(traces),
  candidateDisplay: evaluate(traces, true),
};
if (values.output) writeFileSync(values.output, JSON.stringify(report, null, 2) + '\n');
console.log(
  JSON.stringify(
    {
      runs: report.runs,
      candidate: report.candidate.total,
      candidateDisplay: report.candidateDisplay.total,
    },
    null,
    2,
  ),
);
