import { createServer } from 'node:http';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve, sep } from 'node:path';
import { parseArgs } from 'node:util';
import { chromium, webkit } from 'playwright';

// Opt-in profiling, outside the regular test suite. Never runs browsers concurrently.
const { values } = parseArgs({
  options: {
    browser: { type: 'string', default: 'webkit' },
    reference: { type: 'string' },
    'candidate-root': { type: 'string' },
    candidate: { type: 'string', default: '{}' },
    'reference-options': { type: 'string', default: '{}' },
    repeats: { type: 'string', default: '2' },
    output: { type: 'string' },
    ids: {
      type: 'string',
      default:
        'read-linear,write-scatter-16kb,texture-interp-manual,f32-fma-scalar,f16-fma-mat4,f32-div,workgroup-64,layout-soa,reduction-workgroup,atomic-direct,read-dependent-chain',
    },
  },
});
if (!values.output || !['webkit', 'chromium'].includes(values.browser))
  throw new Error('Provide --output <JSON file> and --browser webkit|chromium');
const roots = {
  reference: resolve(values.reference ?? 'packages/core/dist'),
  candidate: resolve(values['candidate-root'] ?? 'packages/core/dist'),
};
const options = { reference: JSON.parse(values['reference-options']), candidate: JSON.parse(values.candidate) };
const repeats = Number(values.repeats);
if (!Number.isInteger(repeats) || repeats < 1) throw new Error('Invalid --repeats');
const server = createServer((req, res) => {
  if (req.url === '/') {
    res.setHeader('Content-Type', 'text/html');
    res.end('<!doctype html><title>Benchmark profiling</title>');
    return;
  }
  const [, variant, ...parts] = new URL(req.url, 'http://localhost').pathname.split('/');
  const root = roots[variant];
  const file = root && resolve(root, parts.join('/'));
  if (!file || !file.startsWith(root + sep) || !file.endsWith('.js')) {
    res.writeHead(404).end();
    return;
  }
  try {
    res.setHeader('Content-Type', 'text/javascript');
    res.end(readFileSync(file));
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((done) => server.listen(0, '127.0.0.1', done));
let browser;
const report = { browser: values.browser, roots, options, runs: [] };
try {
  browser = await (values.browser === 'webkit' ? webkit : chromium).launch(
    values.browser === 'webkit'
      ? { headless: true }
      : { headless: false, args: ['--headless=new', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] },
  );
  report.version = browser.version();
  for (let repeat = 0; repeat < repeats; repeat++) {
    for (const variant of repeat % 2 ? ['candidate', 'reference'] : ['reference', 'candidate']) {
      const page = await browser.newPage();
      await page.goto(`http://127.0.0.1:${server.address().port}/`);
      const result = await page.evaluate(
        async ({ variant: mode, options: suiteOptions, ids }) => {
          const { runSuite, BENCHMARKS } = await import(`/${mode}/suite.js`);
          const { KernelSampler } = await import(`/${mode}/gpu/benchmarkRunner.js`);

          const phases = { prepareMs: 0, calibrateMs: 0, sampleMs: 0, encodeMs: 0 };
          for (const [method, key] of [
            ['calibrate', 'calibrateMs'],
            ['sample', 'sampleMs'],
          ]) {
            const original = KernelSampler.prototype[method];
            KernelSampler.prototype[method] = async function (...args) {
              const start = performance.now();
              try {
                return await original.apply(this, args);
              } finally {
                phases[key] += performance.now() - start;
              }
            };
          }
          const selected = ids === 'all' ? BENCHMARKS : BENCHMARKS.filter((b) => ids.split(',').includes(b.id));
          if (selected.length !== (ids === 'all' ? BENCHMARKS.length : ids.split(',').length))
            throw new Error('Unknown benchmark ID');
          const benchmarks = selected.map((definition) => ({
            ...definition,
            prepare: async (...args) => {
              const start = performance.now();
              try {
                const prepared = await definition.prepare(...args);
                if (prepared.kind === 'kernel') {
                  const encode = prepared.harness.encode;
                  prepared.harness.encode = function (...encodeArgs) {
                    const encodeStart = performance.now();
                    try {
                      return encode.apply(this, encodeArgs);
                    } finally {
                      // Nested within calibration/sampling, not an additive phase.
                      phases.encodeMs += performance.now() - encodeStart;
                    }
                  };
                }
                return prepared;
              } finally {
                phases.prepareMs += performance.now() - start;
              }
            },
          }));
          const rows = [];
          const completion = [{ at: 0, completed: 0 }];
          const completedIds = new Set();
          const events = [];
          let deviceInfo;
          let maxTimerGapMs = 0;
          let lastTick = performance.now();
          const timer = setInterval(() => {
            const now = performance.now();
            maxTimerGapMs = Math.max(maxTimerGapMs, now - lastTick);
            lastTick = now;
          }, 50);
          const start = performance.now();
          try {
            for await (const row of runSuite({
              minRounds: 3,
              maxRounds: 5,
              ...suiteOptions,
              benchmarks,
              onDeviceInfo: (info) => {
                deviceInfo = info;
              },
              onProgress: (event) => events.push(event),
            })) {
              if (row.status !== 'running') {
                rows.push(row);
                completedIds.add(row.id);
                completion.push({ at: performance.now() - start, completed: completedIds.size });
              }
            }
          } finally {
            clearInterval(timer);
          }
          return {
            elapsedMs: performance.now() - start,
            phases,
            maxTimerGapMs,
            deviceInfo,
            rows,
            events,
            completion,
            benchmarkCount: selected.length,
          };
        },
        { variant, options: options[variant], ids: values.ids },
      );
      report.runs.push({ repeat, variant, ...result });
      writeFileSync(values.output, `${JSON.stringify(report, null, 2)}\n`);
      console.log(
        JSON.stringify({
          repeat,
          variant,
          elapsedMs: result.elapsedMs,
          phases: result.phases,
          maxTimerGapMs: result.maxTimerGapMs,
        }),
      );
      if (result.rows.some((row) => row.status === 'error')) throw new Error('Benchmark error; inspect report');
      await page.close();
      await new Promise((done) => setTimeout(done, 3000));
    }
  }
} finally {
  await browser?.close();
  server.close();
}
