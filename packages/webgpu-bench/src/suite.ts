import { acquireGpuContext, type GpuContext } from './gpu/context.ts';
import { generateMatVecData, padToMultipleOf4 } from './data/generate.ts';
import { KernelSampler } from './gpu/benchmarkRunner.ts';
import { runSampling, type Sampleable, type SampleStateSnapshot } from './sampling.ts';
import { prepareReadBandwidth, prepareWriteBandwidth } from './benchmarks/streamBandwidth.ts';
import {
  prepareFlopsF32Scalar,
  prepareFlopsF32Vec4,
  prepareFlopsF32Mat4,
  prepareFlopsF32Matvec,
} from './benchmarks/flopsF32.ts';
import {
  prepareFlopsF16Scalar,
  prepareFlopsF16Vec4,
  prepareFlopsF16Mat4,
  prepareFlopsF16Matvec,
} from './benchmarks/flopsF16.ts';
import {
  prepareFlopsI8Scalar,
  prepareFlopsI8Vec4,
  prepareFlopsI8Mat4,
  prepareFlopsI8Matvec,
  prepareFlopsI8MatvecDp4a,
  prepareFlopsI8Dp4a,
} from './benchmarks/flopsI8.ts';
import {
  prepareFlopsF32Div,
  prepareFlopsI32Div,
  prepareFlopsF32Sqrt,
  prepareFlopsF32Rsqrt,
  prepareFlopsF32Pow,
  prepareFlopsF32Sincos,
  prepareFlopsF32Log,
} from './benchmarks/flopsMath.ts';
import {
  prepareFlopsU32PackUnpack,
  prepareFlopsI32F32Convert,
  prepareFlopsF32F16Convert,
  prepareFlopsI32F16Convert,
} from './benchmarks/flopsConvert.ts';
import {
  errorResult,
  metricPerSecond,
  rowFromMeta,
  FLOPS_METRIC,
  BYTES_METRIC,
  type BenchmarkMeta,
  type HarnessConfig,
  type PreparedBenchmark,
} from './benchmarks/common.ts';
import type { BenchmarkCategory, BenchmarkResult, SuiteOptions } from './types.ts';

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

const DEFAULT_ROWS = 4096;
const DEFAULT_COLS = 4096;

type BenchmarkEntry = readonly [id: string, category: BenchmarkCategory, prepare: () => Promise<PreparedBenchmark>];
const bandwidth = (id: string, prepare: () => Promise<PreparedBenchmark>): BenchmarkEntry => [id, 'bandwidth', prepare];
const compute = (id: string, prepare: () => Promise<PreparedBenchmark>): BenchmarkEntry => [id, 'compute', prepare];

/** A kernel wired to its sampler, plus the metadata needed to turn timings into a results row. */
interface ScheduledKernel {
  meta: BenchmarkMeta;
  sampler: KernelSampler;
  metaAtWork?: (work: number) => BenchmarkMeta;
}

/**
 * Runs the full benchmark suite, yielding a BenchmarkResult every time a
 * row changes so a UI can render the results table incrementally: once
 * when each benchmark is set up (`running`), after every measurement (still
 * `running`, with the best-so-far), and once when it finishes (ok / skipped
 * / error).
 *
 * Every benchmark isolates a single resource — memory read, memory write,
 * or ALU throughput — rather than modeling a specific real-world op, so
 * results are directly comparable to the device's published bandwidth /
 * FLOPS specs.
 *
 * Measurements are scheduled round-robin across all benchmarks with idle
 * gaps and thermal cooldowns (see `runSampling`), and the reported number
 * for each benchmark is its *best* run — so a phone that throttles partway
 * through the suite still reports what it can do rather than what it was
 * doing while hot.
 */
export async function* runSuite(options: SuiteOptions = {}): AsyncGenerator<BenchmarkResult> {
  const rows = padToMultipleOf4(options.rows ?? DEFAULT_ROWS);
  const cols = padToMultipleOf4(options.cols ?? DEFAULT_COLS);

  const ctx = await acquireGpuContext();
  options.onDeviceInfo?.(ctx.info);

  const data = generateMatVecData(rows, cols);
  const harness: HarnessConfig = {
    targetMs: options.targetMs,
    targetDispatchMs: options.targetDispatchMs,
    warmups: options.warmups,
  };
  const flopsHarness = { ...harness, threads: options.computeThreads, iterations: options.computeIterations };

  const benchmarks: readonly BenchmarkEntry[] = [
    bandwidth('read-bandwidth', () => prepareReadBandwidth(ctx, data, harness)),
    bandwidth('write-bandwidth', () => prepareWriteBandwidth(ctx, data, harness)),
    compute('flops-f32-scalar', () => prepareFlopsF32Scalar(ctx, flopsHarness)),
    compute('flops-f32-vec4', () => prepareFlopsF32Vec4(ctx, flopsHarness)),
    compute('flops-f32-mat4', () => prepareFlopsF32Mat4(ctx, flopsHarness)),
    compute('flops-f32-matvec', () => prepareFlopsF32Matvec(ctx, flopsHarness)),
    compute('flops-f16-scalar', () => prepareFlopsF16Scalar(ctx, flopsHarness)),
    compute('flops-f16-vec4', () => prepareFlopsF16Vec4(ctx, flopsHarness)),
    compute('flops-f16-mat4', () => prepareFlopsF16Mat4(ctx, flopsHarness)),
    compute('flops-f16-matvec', () => prepareFlopsF16Matvec(ctx, flopsHarness)),
    compute('flops-i8-scalar', () => prepareFlopsI8Scalar(ctx, flopsHarness)),
    compute('flops-i8-vec4', () => prepareFlopsI8Vec4(ctx, flopsHarness)),
    compute('flops-i8-mat4', () => prepareFlopsI8Mat4(ctx, flopsHarness)),
    compute('flops-i8-matvec', () => prepareFlopsI8Matvec(ctx, flopsHarness)),
    compute('flops-i8-matvec-dp4a', () => prepareFlopsI8MatvecDp4a(ctx, flopsHarness)),
    compute('flops-i8-dp4a', () => prepareFlopsI8Dp4a(ctx, flopsHarness)),
    compute('flops-f32-div', () => prepareFlopsF32Div(ctx, flopsHarness)),
    compute('flops-i32-div', () => prepareFlopsI32Div(ctx, flopsHarness)),
    compute('flops-f32-sqrt', () => prepareFlopsF32Sqrt(ctx, flopsHarness)),
    compute('flops-f32-rsqrt', () => prepareFlopsF32Rsqrt(ctx, flopsHarness)),
    compute('flops-f32-pow', () => prepareFlopsF32Pow(ctx, flopsHarness)),
    compute('flops-f32-sincos', () => prepareFlopsF32Sincos(ctx, flopsHarness)),
    compute('flops-f32-log', () => prepareFlopsF32Log(ctx, flopsHarness)),
    compute('flops-u32-packunpack', () => prepareFlopsU32PackUnpack(ctx, flopsHarness)),
    compute('flops-i32-f32-convert', () => prepareFlopsI32F32Convert(ctx, flopsHarness)),
    compute('flops-f32-f16-convert', () => prepareFlopsF32F16Convert(ctx, flopsHarness)),
    compute('flops-i32-f16-convert', () => prepareFlopsI32F16Convert(ctx, flopsHarness)),
  ];

  // Phase 1: build every benchmark's GPU resources up front. Rows that can't
  // run at all (missing feature, failed shader compile) resolve right here;
  // the rest show up as `running` so the table has its final shape before
  // the first measurement lands.
  const scheduled: ScheduledKernel[] = [];
  for (const [id, category, prepare] of benchmarks) {
    const fallbackMeta: BenchmarkMeta = {
      id,
      label: id,
      description: '',
      source: '',
      category,
      rows,
      cols,
      metric: category === 'bandwidth' ? BYTES_METRIC : FLOPS_METRIC,
      amountPerOp: 0,
    };
    let prepared: PreparedBenchmark;
    try {
      prepared = await prepare();
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
