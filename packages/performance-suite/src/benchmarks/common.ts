import type { GpuContext } from '../gpu/context.ts';
import { runTimedBenchmark } from '../gpu/benchmarkRunner.ts';
import type { BenchmarkCategory, BenchmarkResult } from '../types.ts';

export function createPipeline(
  device: GPUDevice,
  label: string,
  code: string,
  constants?: Record<string, number>,
): GPUComputePipeline {
  const module = device.createShaderModule({ label, code });
  return device.createComputePipeline({
    label,
    layout: 'auto',
    compute: { module, entryPoint: 'main', constants },
  });
}

export interface MatVecFlopStats {
  rows: number;
  cols: number;
  /** Total bytes moved to/from GPU memory per op (matrix + vector, at their storage precision). */
  bytesPerOp: number;
  /**
   * Total FLOPs for one op, overriding the default matvec MAC convention
   * (`2 * rows * cols`) — used by benchmarks that aren't shaped like a
   * matvec (bandwidth streams, raw-FLOPS compute kernels).
   */
  flopsOverride?: number;
}

export function flopsAndBandwidth(stats: MatVecFlopStats, meanMs: number): { gflops: number; gbps: number } {
  const seconds = meanMs / 1000;
  const flops = stats.flopsOverride ?? 2 * stats.rows * stats.cols;
  return {
    gflops: flops / seconds / 1e9,
    gbps: stats.bytesPerOp / seconds / 1e9,
  };
}

export interface HarnessConfig {
  targetMs?: number;
  warmups?: number;
  runs?: number;
}

export interface RunKernelOptions {
  id: string;
  label: string;
  description: string;
  category: BenchmarkCategory;
  ctx: GpuContext;
  rows: number;
  cols: number;
  bytesPerOp: number;
  /** See {@link MatVecFlopStats.flopsOverride}. */
  flopsOverride?: number;
  workgroupsPerIteration: [number, number, number];
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  targetMs?: number;
  warmups?: number;
  runs?: number;
}

/** Runs the calibrated warmup+timed-measurement harness for a single matvec kernel and packages a BenchmarkResult. */
export async function runKernelBenchmark(opts: RunKernelOptions): Promise<BenchmarkResult> {
  const { ctx, pipeline, bindGroup, workgroupsPerIteration } = opts;
  const timed = await runTimedBenchmark({
    device: ctx.device,
    useTimestamps: ctx.info.supportsTimestampQuery,
    targetMs: opts.targetMs,
    warmups: opts.warmups,
    runs: opts.runs,
    encode: (pass, iterations) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let i = 0; i < iterations; i++) {
        pass.dispatchWorkgroups(...workgroupsPerIteration);
      }
    },
  });

  const { gflops, gbps } = flopsAndBandwidth(
    { rows: opts.rows, cols: opts.cols, bytesPerOp: opts.bytesPerOp, flopsOverride: opts.flopsOverride },
    timed.stats.mean,
  );

  return {
    id: opts.id,
    label: opts.label,
    description: opts.description,
    category: opts.category,
    status: 'ok',
    rows: opts.rows,
    cols: opts.cols,
    innerIterations: timed.innerIterations,
    timesMs: timed.timesMs,
    stats: timed.stats,
    gflops,
    gbps,
    timingMethod: timed.timingMethod,
  };
}

export function skippedResult(
  id: string,
  label: string,
  description: string,
  category: BenchmarkCategory,
  rows: number,
  cols: number,
  message: string,
): BenchmarkResult {
  return {
    id,
    label,
    description,
    category,
    status: 'skipped',
    message,
    rows,
    cols,
    innerIterations: 0,
    timesMs: [],
    timingMethod: 'cpu-wallclock',
  };
}

export function errorResult(
  id: string,
  label: string,
  description: string,
  category: BenchmarkCategory,
  rows: number,
  cols: number,
  error: unknown,
): BenchmarkResult {
  return {
    id,
    label,
    description,
    category,
    status: 'error',
    message: error instanceof Error ? error.message : String(error),
    rows,
    cols,
    innerIterations: 0,
    timesMs: [],
    timingMethod: 'cpu-wallclock',
  };
}
