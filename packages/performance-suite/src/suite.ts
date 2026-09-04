import { acquireGpuContext } from './gpu/context.ts';
import { generateMatVecData, padToMultipleOf4 } from './data/generate.ts';
import { benchmarkReadBandwidth, benchmarkWriteBandwidth } from './benchmarks/streamBandwidth.ts';
import {
  benchmarkFlopsF32Scalar,
  benchmarkFlopsF32Vec4,
  benchmarkFlopsF32Mat4,
  benchmarkFlopsF32Matvec,
} from './benchmarks/flopsF32.ts';
import {
  benchmarkFlopsF16Scalar,
  benchmarkFlopsF16Vec4,
  benchmarkFlopsF16Mat4,
  benchmarkFlopsF16Matvec,
} from './benchmarks/flopsF16.ts';
import {
  benchmarkFlopsI8Scalar,
  benchmarkFlopsI8Vec4,
  benchmarkFlopsI8Mat4,
  benchmarkFlopsI8Matvec,
  benchmarkFlopsI8MatvecDp4a,
  benchmarkFlopsI8Dp4a,
} from './benchmarks/flopsI8.ts';
import { errorResult, type HarnessConfig } from './benchmarks/common.ts';
import type { BenchmarkResult, SuiteOptions } from './types.ts';

export type {
  BenchmarkResult,
  SuiteOptions,
  DeviceInfo,
  Stats,
  BenchmarkCategory,
  BenchmarkStatus,
  SamplingStopReason,
} from './types.ts';
export { computeStats } from './stats.ts';

const DEFAULT_ROWS = 4096;
const DEFAULT_COLS = 4096;

/**
 * Runs the full benchmark suite, yielding a BenchmarkResult as soon as each
 * benchmark finishes (ok, skipped, or error) so a UI can render the results
 * table incrementally. Every benchmark isolates a single resource — memory
 * read, memory write, or ALU throughput — rather than modeling a specific
 * real-world op, so results are directly comparable to the device's
 * published bandwidth/FLOPS specs.
 */
export async function* runSuite(options: SuiteOptions = {}): AsyncGenerator<BenchmarkResult> {
  const rows = padToMultipleOf4(options.rows ?? DEFAULT_ROWS);
  const cols = padToMultipleOf4(options.cols ?? DEFAULT_COLS);
  const computeThreads = options.computeThreads;
  const computeIterations = options.computeIterations;

  const ctx = await acquireGpuContext();
  options.onDeviceInfo?.(ctx.info);

  const data = generateMatVecData(rows, cols);
  const harness: HarnessConfig = {
    targetMs: options.targetMs,
    warmups: options.warmups,
    minRuns: options.minRuns,
    maxRuns: options.maxRuns,
    precision: options.precision,
  };
  const flopsHarness = { ...harness, threads: computeThreads, iterations: computeIterations };

  const bandwidthBenchmarks: Array<[string, () => Promise<BenchmarkResult>]> = [
    ['read-bandwidth', () => benchmarkReadBandwidth(ctx, data, harness)],
    ['write-bandwidth', () => benchmarkWriteBandwidth(ctx, data, harness)],
  ];

  for (const [id, run] of bandwidthBenchmarks) {
    try {
      yield await run();
    } catch (error) {
      yield errorResult(id, id, '', 'bandwidth', rows, cols, error);
    }
  }

  const flopsBenchmarks: Array<[string, () => Promise<BenchmarkResult>]> = [
    ['flops-f32-scalar', () => benchmarkFlopsF32Scalar(ctx, flopsHarness)],
    ['flops-f32-vec4', () => benchmarkFlopsF32Vec4(ctx, flopsHarness)],
    ['flops-f32-mat4', () => benchmarkFlopsF32Mat4(ctx, flopsHarness)],
    ['flops-f32-matvec', () => benchmarkFlopsF32Matvec(ctx, flopsHarness)],
    ['flops-f16-scalar', () => benchmarkFlopsF16Scalar(ctx, flopsHarness)],
    ['flops-f16-vec4', () => benchmarkFlopsF16Vec4(ctx, flopsHarness)],
    ['flops-f16-mat4', () => benchmarkFlopsF16Mat4(ctx, flopsHarness)],
    ['flops-f16-matvec', () => benchmarkFlopsF16Matvec(ctx, flopsHarness)],
    ['flops-i8-scalar', () => benchmarkFlopsI8Scalar(ctx, flopsHarness)],
    ['flops-i8-vec4', () => benchmarkFlopsI8Vec4(ctx, flopsHarness)],
    ['flops-i8-mat4', () => benchmarkFlopsI8Mat4(ctx, flopsHarness)],
    ['flops-i8-matvec', () => benchmarkFlopsI8Matvec(ctx, flopsHarness)],
    ['flops-i8-matvec-dp4a', () => benchmarkFlopsI8MatvecDp4a(ctx, flopsHarness)],
    ['flops-i8-dp4a', () => benchmarkFlopsI8Dp4a(ctx, flopsHarness)],
  ];

  for (const [id, run] of flopsBenchmarks) {
    try {
      yield await run();
    } catch (error) {
      yield errorResult(id, id, '', 'compute', rows, cols, error);
    }
  }
}
