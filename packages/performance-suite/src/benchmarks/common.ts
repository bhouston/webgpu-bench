import type { GpuContext } from '../gpu/context.ts';
import { runTimedBenchmark, type SamplingConfig } from '../gpu/benchmarkRunner.ts';
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
 * misleadingly "ok" benchmark result with 0s mean/stddev, rather than the
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

/** Sampling knobs (`targetMs`, `warmups`, `minRuns`, `maxRuns`, `precision`) shared by every benchmark. */
export type HarnessConfig = SamplingConfig;

export interface RunKernelOptions extends HarnessConfig {
  id: string;
  label: string;
  description: string;
  category: BenchmarkCategory;
  ctx: GpuContext;
  rows: number;
  cols: number;
  bytes: number;
  flops: number;
  workgroupsPerIteration: [number, number, number];
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
}

/** Runs the calibrated warmup + adaptive-sampling harness for a single kernel and packages a BenchmarkResult. */
export async function runKernelBenchmark(opts: RunKernelOptions): Promise<BenchmarkResult> {
  const { ctx, pipeline, bindGroup, workgroupsPerIteration } = opts;
  const timed = await runTimedBenchmark({
    device: ctx.device,
    useTimestamps: ctx.info.supportsTimestampQuery,
    targetMs: opts.targetMs,
    warmups: opts.warmups,
    minRuns: opts.minRuns,
    maxRuns: opts.maxRuns,
    precision: opts.precision,
    encode: (pass, iterations) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let i = 0; i < iterations; i++) {
        pass.dispatchWorkgroups(...workgroupsPerIteration);
      }
    },
  });

  // A mean of exactly 0 isn't a real timing: it means every readback of the
  // (zero-initialized) GPU-timestamp buffer came back unwritten, which
  // happens when the compute pass silently did no work — e.g. an invalid
  // pipeline/bind group that errors out post-submission rather than at
  // pipeline-creation time. Surface that as a failure instead of an "ok"
  // result with a nonsensical 0s/Infinity-throughput row.
  if (timed.stats.mean <= 0) {
    throw new Error(
      `"${opts.label}" measured a mean time of ${timed.stats.mean}ms across ${timed.timesMs.length} runs — the GPU pass likely did no work (check for an 'uncapturederror' in the console).`,
    );
  }

  // Throughput is derived from the median rather than the mean: a single
  // slow run (GPU clock still ramping, a background compositor frame) skews
  // the mean but leaves the median untouched.
  const { gflops, gbps } = flopsAndBandwidth({ flops: opts.flops, bytes: opts.bytes }, timed.stats.median);

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
    stopReason: timed.stopReason,
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
