import { acquireGpuContext, type GpuContext } from './gpu/context.ts';
import { generateMatVecData, padToMultipleOf4 } from './data/generate.ts';
import { KernelSampler } from './gpu/benchmarkRunner.ts';
import { runSampling, type Sampleable, type SampleStateSnapshot } from './sampling.ts';
import {
  errorResult,
  metricPerSecond,
  rowFromMeta,
  type BenchmarkContext,
  type BenchmarkDefinition,
  type BenchmarkMeta,
  type HarnessConfig,
  type PreparedBenchmark,
} from './benchmarks/common.ts';
import { BENCHMARKS } from './catalog.ts';
import type { BenchmarkResult, SuiteOptions } from './types.ts';

export type {
  BenchmarkResult,
  SuiteOptions,
  DeviceInfo,
  Stats,
  BenchmarkCategory,
  BenchmarkStatus,
  SamplingStopReason,
  SuiteProgressEvent,
  MetricDef,
} from './types.ts';
export { computeStats } from './stats.ts';
export { BENCHMARKS } from './catalog.ts';
export type { BenchmarkContext, BenchmarkDefinition } from './benchmarks/common.ts';

const DEFAULT_ROWS = 4096;
const DEFAULT_COLS = 4096;

/** A benchmark wired to its sampler, plus the metadata needed to turn timings into a results row. */
interface ScheduledKernel {
  meta: BenchmarkMeta;
  sampler: KernelSampler;
  metaAtWork?: (work: number) => BenchmarkMeta;
}

/**
 * Runs a suite of benchmarks — this package's own (`BENCHMARKS`, the
 * default) unless `options.benchmarks` names a different list — yielding a
 * BenchmarkResult every time a row changes so a UI can render the results
 * table incrementally: once when each benchmark is set up (`running`), after
 * every measurement (still `running`, with the best-so-far), and once when
 * it finishes (ok / skipped / error).
 *
 * `runSuite` itself knows nothing about what any given benchmark measures —
 * it just calls each `BenchmarkDefinition.prepare` (see `benchmarks/common.ts`)
 * to get GPU resources, then schedules and times them. Bring your own
 * definitions, mix them with the built-ins, or run a filtered subset of
 * `BENCHMARKS` — the scheduling below doesn't change.
 *
 * Measurements are scheduled round-robin across all benchmarks with idle
 * gaps and thermal cooldowns (see `runSampling`), and the reported number
 * for each benchmark is its *best* run — so a phone that throttles partway
 * through the suite still reports what it can do rather than what it was
 * doing while hot.
 */
export async function* runSuite(options: SuiteOptions = {}): AsyncGenerator<BenchmarkResult> {
  const benchmarks: readonly BenchmarkDefinition[] = options.benchmarks ?? BENCHMARKS;
  const rows = padToMultipleOf4(options.rows ?? DEFAULT_ROWS);
  const cols = padToMultipleOf4(options.cols ?? DEFAULT_COLS);

  const ctx = await acquireGpuContext();
  options.onDeviceInfo?.(ctx.info);

  const harness: HarnessConfig = {
    targetMs: options.targetMs,
    targetDispatchMs: options.targetDispatchMs,
    warmups: options.warmups,
  };
  const bc: BenchmarkContext = {
    ctx,
    data: generateMatVecData(rows, cols),
    harness,
    computeThreads: options.computeThreads,
    computeIterations: options.computeIterations,
  };

  // Phase 1: build every benchmark's GPU resources up front. Rows that can't
  // run at all (missing feature, failed shader compile) resolve right here;
  // the rest show up as `running` so the table has its final shape before
  // the first measurement lands.
  const scheduled: ScheduledKernel[] = [];
  for (const def of benchmarks) {
    const fallbackMeta: BenchmarkMeta = { id: def.id, category: def.category, rows, cols, amountPerOp: 0 };
    let prepared: PreparedBenchmark;
    try {
      prepared = await def.prepare(bc);
    } catch (error) {
      yield errorResult(fallbackMeta, error);
      continue;
    }
    if (prepared.kind === 'skipped') {
      yield prepared.result;
      continue;
    }
    scheduled.push({
      meta: prepared.meta,
      sampler: new KernelSampler(prepared.harness),
      metaAtWork: prepared.metaAtWork,
    });
    yield rowFromMeta(prepared.meta);
  }

  // Phase 2: sample everything round-robin. `runSampling` pushes updates
  // through a callback; bridge them into this generator via a queue.
  const byId = new Map(scheduled.map((k) => [k.meta.id, k]));
  const queue: BenchmarkResult[] = [];
  let wake: (() => void) | null = null;
  const push = (r: BenchmarkResult) => {
    queue.push(r);
    wake?.();
    wake = null;
  };

  const sampleables: Sampleable[] = scheduled.map(({ meta, sampler }) => ({
    id: meta.id,
    calibrate: () => sampler.calibrate(),
    sample: () => sampler.sample(),
  }));

  const run = runSampling(sampleables, {
    ...pickSamplingOptions(options),
    onProgress: options.onProgress,
    onUpdate: (state) => push(resultFromState(byId.get(state.id)!, state)),
  }).finally(() => {
    for (const k of scheduled) k.sampler.destroy();
  });
  let done = false;
  void run.then(
    () => {
      done = true;
      wake?.();
    },
    () => {
      done = true;
      wake?.();
    },
  );

  try {
    while (true) {
      while (queue.length > 0) yield queue.shift()!;
      if (done) break;
      await new Promise<void>((resolve) => {
        wake = resolve;
      });
    }
    await run;
  } finally {
    ctx.device.destroy();
  }
}

function pickSamplingOptions(o: SuiteOptions) {
  return {
    minRounds: o.minRounds,
    maxRounds: o.maxRounds,
    stableRounds: o.stableRounds,
    improvementTolerance: o.improvementTolerance,
    throttleThreshold: o.throttleThreshold,
    idleMs: o.idleMs,
    cooldownMs: o.cooldownMs,
    maxCooldowns: o.maxCooldowns,
    throttledFraction: o.throttledFraction,
  };
}

function resultFromState(kernel: ScheduledKernel, state: SampleStateSnapshot): BenchmarkResult {
  const { sampler } = kernel;
  // Once calibrated, the problem size (and so FLOPs per op) is whatever the sampler settled on.
  const meta = kernel.metaAtWork && sampler.work !== undefined ? kernel.metaAtWork(sampler.work) : kernel.meta;
  if (state.error !== undefined) return errorResult(meta, state.error);

  const row = rowFromMeta(meta);
  row.status = state.stopReason ? 'ok' : 'running';
  row.innerIterations = sampler.innerIterations;
  row.timingMethod = sampler.timingMethod;
  row.timesMs = state.timesMs;
  row.throttledMs = state.throttledMs;
  row.stats = state.stats;
  row.stopReason = state.stopReason;
  if (state.stats) {
    // Headline number comes from the best run: every noise source only ever
    // slows a run down, so the minimum is the least-contaminated estimate.
    row.metricValue = metricPerSecond(meta.amountPerOp, state.stats.min);
  }
  if (state.stopReason === 'throttled') {
    row.message = 'Device stayed thermally throttled through every cooldown; best run may understate it.';
  }
  return row;
}

export type { GpuContext };
