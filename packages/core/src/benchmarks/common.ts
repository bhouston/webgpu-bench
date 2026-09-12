import type { GpuContext } from '../gpu/context.ts';
import type { KernelHarness, MeasurementConfig, WorkKnob } from '../gpu/benchmarkRunner.ts';
import type { GeneratedData } from '../data/generate.ts';
import type { BenchmarkCategory, BenchmarkResult, MetricDef } from '../types.ts';

/** Shared metric definitions. `key` is what a JSON export would key the value by, stable across every benchmark that reports it. */
export const FLOPS_METRIC: MetricDef = { key: 'flops', unit: 'FLOP', name: 'Floating-point ops' };
export const OPS_METRIC: MetricDef = { key: 'ops', unit: 'OP', name: 'Operations' };
export const BYTES_METRIC: MetricDef = { key: 'bytes', unit: 'B', name: 'Bandwidth' };

/**
 * Compiles a shader module and builds a compute pipeline from it, watching
 * for validation errors via error scopes rather than trusting the WebGPU
 * calls to throw. Both `createShaderModule` and `createComputePipeline`
 * (`'auto'` layout) succeed synchronously even when the WGSL fails to
 * compile or the pipeline is otherwise invalid — the failure only surfaces
 * later as a device-level 'uncapturederror' when something dispatches
 * against it. Left unchecked, that means the compute pass silently does
 * nothing: the command buffer still submits and completes almost
 * instantly, so a GPU-timestamp read of a never-touched, zero-initialized
 * query buffer comes back as exactly 0ns elapsed — reported as a
 * misleadingly "ok" benchmark result with 0s timings, rather than the
 * shader-compile failure it actually is.
 */
export async function createPipeline(
  device: GPUDevice,
  label: string,
  code: string,
  constants?: Record<string, number>,
  entryPoint = 'main',
): Promise<GPUComputePipeline> {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ label, code });
  const pipeline = device.createComputePipeline({
    label,
    layout: 'auto',
    compute: { module, entryPoint, constants },
  });
  const error = await device.popErrorScope();
  if (error) {
    throw new Error(`Failed to create pipeline "${label}": ${error.message}`);
  }
  return pipeline;
}

/** `<metric.unit>/s`, from the amount of that unit moved/computed in one op and that op's best time. */
export function metricPerSecond(amountPerOp: number, perOpMs: number): number {
  return amountPerOp / (perOpMs / 1000);
}

/** Per-measurement knobs (`targetMs`, `targetDispatchMs`, `warmups`) shared by every benchmark. */
export type HarnessConfig = MeasurementConfig;

/**
 * Everything about a benchmark that's known before it's measured. Identity/display
 * fields (label, description, source, metric) live on its `BenchmarkDefinition`
 * instead — this is just what a live run needs to size and account for the work.
 */
export interface BenchmarkMeta {
  id: string;
  category: BenchmarkCategory;
  rows: number;
  cols: number;
  /** Amount of `metric.unit` (per the catalog's `MetricDef` for this id) moved/computed per op. */
  amountPerOp: number;
}

/**
 * A benchmark whose GPU resources are built and which is ready to be
 * measured by the scheduler, or one that has already resolved to a
 * `skipped` row (missing device feature).
 */
export type PreparedBenchmark =
  | {
      kind: 'kernel';
      meta: BenchmarkMeta;
      harness: KernelHarness;
      /** For kernels with a work knob: the metadata (problem size, FLOPs, bytes per op) at a given work setting. */
      metaAtWork?: (work: number) => BenchmarkMeta;
    }
  | { kind: 'skipped'; result: BenchmarkResult };

export interface PrepareKernelOptions extends HarnessConfig, BenchmarkMeta {
  ctx: GpuContext;
  workgroupsPerIteration: [number, number, number];
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  work?: WorkKnob;
  metaAtWork?: (work: number) => BenchmarkMeta;
}

/** Packages a built pipeline/bind group as a kernel the sampling scheduler can measure. */
export function prepareKernelBenchmark(opts: PrepareKernelOptions): PreparedBenchmark {
  const { ctx, pipeline, bindGroup, workgroupsPerIteration } = opts;
  return {
    kind: 'kernel',
    meta: {
      id: opts.id,
      category: opts.category,
      rows: opts.rows,
      cols: opts.cols,
      amountPerOp: opts.amountPerOp,
    },
    metaAtWork: opts.metaAtWork,
    harness: {
      device: ctx.device,
      useTimestamps: ctx.info.supportsTimestampQuery,
      targetMs: opts.targetMs,
      targetDispatchMs: opts.targetDispatchMs,
      warmups: opts.warmups,
      maxIterations: opts.maxIterations,
      work: opts.work,
      encode: (pass, iterations) => {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        for (let i = 0; i < iterations; i++) {
          pass.dispatchWorkgroups(...workgroupsPerIteration);
        }
      },
    },
  };
}

/** What `runSuite` hands a `BenchmarkDefinition.prepare` to build its GPU resources. */
export interface BenchmarkContext {
  ctx: GpuContext;
  /** Deterministic matrix/vector data, sized per `SuiteOptions.rows`/`cols`. Only the bandwidth kernels use it. */
  data: GeneratedData;
  harness: HarnessConfig;
  /** See `SuiteOptions.computeThreads`. Only the raw-FLOPS compute kernels use this. */
  computeThreads?: number;
  /** See `SuiteOptions.computeIterations`. Only the raw-FLOPS compute kernels use this. */
  computeIterations?: number;
}

/**
 * A benchmark, fully self-contained: display metadata plus how to build it.
 * This is the contract `runSuite` runs against — the built-in benchmarks in
 * `catalog.ts` implement it, and so can anyone else's, passed in via
 * `SuiteOptions.benchmarks` alongside, instead of, or as a filtered subset
 * of the built-ins.
 */
export interface BenchmarkDefinition {
  id: string;
  label: string;
  description: string;
  /** WGSL source of the kernel, for display alongside a result. */
  source: string;
  category: BenchmarkCategory;
  metric: MetricDef;
  prepare(bc: BenchmarkContext): Promise<PreparedBenchmark>;
}

export function skippedResult(meta: BenchmarkMeta, message: string): BenchmarkResult {
  return {
    ...rowFromMeta(meta),
    status: 'skipped',
    message,
  };
}

export function errorResult(meta: BenchmarkMeta, error: unknown): BenchmarkResult {
  return {
    ...rowFromMeta(meta),
    status: 'error',
    message: error instanceof Error ? error.message : String(error),
  };
}

/** A blank row for a benchmark: identity + problem size, no timings yet. */
export function rowFromMeta(meta: BenchmarkMeta): BenchmarkResult {
  return {
    id: meta.id,
    category: meta.category,
    status: 'running',
    rows: meta.rows,
    cols: meta.cols,
    innerIterations: 0,
    timesMs: [],
    throttledMs: [],
    timingMethod: 'cpu-wallclock',
  };
}
