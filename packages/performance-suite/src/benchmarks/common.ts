import type { GpuContext } from '../gpu/context.ts';
import type { KernelHarness, MeasurementConfig, WorkKnob } from '../gpu/benchmarkRunner.ts';
import type { BenchmarkCategory, BenchmarkResult } from '../types.ts';

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
): Promise<GPUComputePipeline> {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ label, code });
  const pipeline = device.createComputePipeline({
    label,
    layout: 'auto',
    compute: { module, entryPoint: 'main', constants },
  });
  const error = await device.popErrorScope();
  if (error) {
    throw new Error(`Failed to create pipeline "${label}": ${error.message}`);
  }
  return pipeline;
}

/** Total FLOPs and bytes moved for one op — each benchmark computes its own, since bandwidth kernels are ~all bytes and compute kernels are ~all FLOPs. */
export interface ThroughputStats {
  flops: number;
  bytes: number;
}

export function flopsAndBandwidth(stats: ThroughputStats, perOpMs: number): { gflops: number; gbps: number } {
  const seconds = perOpMs / 1000;
  return {
    gflops: stats.flops / seconds / 1e9,
    gbps: stats.bytes / seconds / 1e9,
  };
}

/** Per-measurement knobs (`targetMs`, `targetDispatchMs`, `warmups`) shared by every benchmark. */
export type HarnessConfig = MeasurementConfig;

/** Everything about a benchmark that's known before it's measured. */
export interface BenchmarkMeta {
  id: string;
  label: string;
  description: string;
  category: BenchmarkCategory;
  rows: number;
  cols: number;
  /** Bytes moved per op, for the bandwidth number. */
  bytes: number;
  /** FLOPs per op, for the throughput number. */
  flops: number;
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
      label: opts.label,
      description: opts.description,
      category: opts.category,
      rows: opts.rows,
      cols: opts.cols,
      bytes: opts.bytes,
      flops: opts.flops,
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
    label: meta.label,
    description: meta.description,
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
