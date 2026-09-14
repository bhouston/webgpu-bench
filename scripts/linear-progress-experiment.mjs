// Research-only scheduler variant. Build core first; never edits the production build.
import { cpSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { gunzipSync } from 'node:zlib';

const { values } = parseArgs({
  options: {
    prepare: { type: 'string' },
    report: { type: 'string', multiple: true },
    output: { type: 'string' },
  },
});

if (values.prepare) {
  const target = resolve(values.prepare);
  const source = resolve('packages/core/dist');
  if (target === source || source.startsWith(target + '/')) throw new Error('Use a separate temporary directory');
  mkdirSync(target, { recursive: true });
  cpSync(source, target, { recursive: true });
  const suitePath = `${target}/suite.js`;
  const suite = readFileSync(suitePath, 'utf8');
  if (!suite.includes('import { runSampling } from "./sampling.js";')) throw new Error('Unexpected suite import');
  writeFileSync(
    suitePath,
    suite.replace(
      'import { runSampling } from "./sampling.js";',
      'import { runSampling } from "./linear-sampling.js";',
    ),
  );
  const samplingPath = `${target}/sampling.js`;
  const sampling = readFileSync(samplingPath, 'utf8');
  if (!sampling.includes('const realSleep =')) throw new Error('Unexpected sleep implementation');
  writeFileSync(samplingPath, sampling.replace('const realSleep =', 'export const realSleep ='));
  writeFileSync(
    `${target}/linear-sampling.js`,
    `
import { recordSample, resolveSamplingConfig, realSleep } from './sampling.js';
import { computeStats } from './stats.js';
// Controlled sequential trial: same calibration, kept-sample rule and sample/idle
// durations. Two consecutive discards replace cross-kernel thermal consensus.
// Per-kernel cooldown budgets avoid aborting benchmarks not yet measured.
export async function runSampling(benchmarks, options = {}) {
  const cfg = resolveSamplingConfig(options);
  const sleep = options.sleep ?? realSleep;
  const results = new Map();
  let sampled = false;
  for (const b of benchmarks) {
    const state = { id: b.id, timesMs: [], throttledMs: [], bestMs: Infinity };
    const update = async () => {
      const value = { ...state, timesMs: [...state.timesMs], throttledMs: [...state.throttledMs],
        stats: state.timesMs.length ? computeStats(state.timesMs) : undefined };
      results.set(b.id, value);
      await options.onUpdate?.(value);
    };
    let cooldowns = 0, consecutiveDiscards = 0;
    try {
      await b.calibrate();
      while (!state.stopReason) {
        if (sampled && cfg.idleMs > 0) await sleep(cfg.idleMs);
        sampled = true;
        const ms = await b.sample();
        if (!Number.isFinite(ms) || ms <= 0) throw new Error('Invalid sample duration');
        const kept = recordSample(state, ms, cfg);
        consecutiveDiscards = kept ? 0 : consecutiveDiscards + 1;
        await update();
        if (!state.stopReason && consecutiveDiscards >= 2) {
          if (cooldowns >= cfg.maxCooldowns) {
            state.stopReason = 'throttled';
            options.onProgress?.({ type: 'throttle-abort', throttledIds: [b.id] });
            await update();
            break;
          }
          options.onProgress?.({ type: 'cooldown', attempt: ++cooldowns, maxAttempts: cfg.maxCooldowns,
            ms: cfg.cooldownMs, throttledIds: [b.id] });
          await sleep(cfg.cooldownMs);
          consecutiveDiscards = 0;
        }
      }
    } catch (error) {
      state.error = error;
      await update();
    }
  }
  return results;
}
`,
  );
  console.log(target);
}

function metrics(run) {
  let index = 0,
    visible = 0,
    good = 0,
    error = 0;
  // Count is exact as work completion. This additionally tests whether a user
  // could interpret completed/count as a wall-time percentage within 20%.
  const step = Math.min(50, run.elapsedMs / 1000);
  for (let at = step / 2; at < run.elapsedMs; at += step) {
    while (index + 1 < run.completion.length && run.completion[index + 1].at <= at) index++;
    const completed = run.completion[index].completed;
    if (completed === 0) continue;
    const estimate = completed / run.benchmarkCount;
    const relativeError = Math.abs(estimate / (at / run.elapsedMs) - 1);
    visible += step;
    good += relativeError <= 0.2 ? step : 0;
    error += relativeError * step;
  }
  return {
    coverage: (100 * visible) / run.elapsedMs,
    within20: visible ? (100 * good) / visible : null,
    meanRelativeError: visible ? (100 * error) / visible : null,
    firstResultPercent: (100 * (run.completion[1]?.at ?? run.elapsedMs)) / run.elapsedMs,
  };
}
const median = (numbers) => {
  const sorted = numbers.toSorted((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
if (values.report) {
  const reports = values.report.map((file) => {
    const report = JSON.parse(
      file.endsWith('.gz') ? gunzipSync(readFileSync(file)).toString('utf8') : readFileSync(file, 'utf8'),
    );
    const variants = Object.fromEntries(
      ['reference', 'candidate'].map((variant) => {
        const runs = report.runs.filter((r) => r.variant === variant);
        const scores = Object.fromEntries(
          runs[0].rows
            .filter((r) => r.status === 'ok')
            .map((r) => [
              r.id,
              median(runs.map((run) => run.rows.find((row) => row.id === r.id)?.metricValue).filter(Number.isFinite)),
            ]),
        );
        return [
          variant,
          {
            medianSeconds: median(runs.map((r) => r.elapsedMs / 1000)),
            scores,
            runs: runs.map((r) => ({
              seconds: r.elapsedMs / 1000,
              countProgress: metrics(r),
              cooldowns: r.events.filter((e) => e.type === 'cooldown').length,
              discarded: r.rows.reduce((n, row) => n + row.throttledMs.length, 0),
              stopReasons: r.rows.map((row) => ({
                id: row.id,
                reason: row.stopReason,
                kept: row.timesMs.length,
                timingMethod: row.timingMethod,
              })),
            })),
          },
        ];
      }),
    );
    const scoreChanges = Object.fromEntries(
      Object.entries(variants.reference.scores).map(([id, score]) => [
        id,
        100 * (variants.candidate.scores[id] / score - 1),
      ]),
    );
    return { file, browser: report.browser, ...variants, scoreChanges };
  });
  const output = JSON.stringify(reports, null, 2);
  if (values.output) writeFileSync(values.output, output + '\n');
  console.log(output);
}
