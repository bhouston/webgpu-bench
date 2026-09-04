import { acquireGpuContext } from './gpu/context.ts';
import { generateMatVecData, padToMultipleOf4 } from './data/generate.ts';
import { benchmarkF32Scalar } from './benchmarks/matvecF32Scalar.ts';
import { benchmarkF32Vec4 } from './benchmarks/matvecF32Vec4.ts';
import { benchmarkF32Vec4Unrolled } from './benchmarks/matvecF32Vec4Unrolled.ts';
import { benchmarkF32Vec4Shared } from './benchmarks/matvecF32Vec4Shared.ts';
import { benchmarkF16Scalar, benchmarkF16Vec4 } from './benchmarks/matvecF16.ts';
import { benchmarkI8Packed } from './benchmarks/matvecI8Packed.ts';
import { runWorkgroupSweep } from './benchmarks/workgroupSweep.ts';
import { benchmarkMlpMultiLayer } from './benchmarks/mlpMultiLayer.ts';
import { benchmarkReadBandwidth, benchmarkWriteBandwidth } from './benchmarks/streamBandwidth.ts';
import { benchmarkFlopsF32Scalar, benchmarkFlopsF32Vec4, benchmarkFlopsF32Mat4 } from './benchmarks/flopsF32.ts';
import { benchmarkFlopsF16Scalar, benchmarkFlopsF16Vec4, benchmarkFlopsF16Mat4 } from './benchmarks/flopsF16.ts';
import { benchmarkFlopsI8Scalar, benchmarkFlopsI8Vec4, benchmarkFlopsI8Mat4, benchmarkFlopsI8Dp4a } from './benchmarks/flopsI8.ts';
import { errorResult } from './benchmarks/common.ts';
import type { BenchmarkResult, SuiteOptions } from './types.ts';

export type { BenchmarkResult, SuiteOptions, DeviceInfo, Stats, BenchmarkCategory, BenchmarkStatus } from './types.ts';
export { computeStats } from './stats.ts';

const DEFAULT_ROWS = 4096;
const DEFAULT_COLS = 4096;
const DEFAULT_MLP_LAYERS = [4096, 4096, 4096, 4096];

/**
 * Runs the full benchmark suite, yielding a BenchmarkResult as soon as each
 * benchmark finishes (ok, skipped, or error) so a UI can render the results
 * table incrementally. All matrix/vector data is generated deterministically
 * from a fixed seed, so results are directly comparable run to run.
 */
export async function* runSuite(options: SuiteOptions = {}): AsyncGenerator<BenchmarkResult> {
  const rows = padToMultipleOf4(options.rows ?? DEFAULT_ROWS);
  const cols = padToMultipleOf4(options.cols ?? DEFAULT_COLS);
  const targetMs = options.targetMs ?? 300;
  const warmups = options.warmups ?? 3;
  const runs = options.runs ?? 10;
  const mlpLayers = options.mlpLayers ?? DEFAULT_MLP_LAYERS;
  const computeThreads = options.computeThreads;
  const computeIterations = options.computeIterations;

  const ctx = await acquireGpuContext();
  options.onDeviceInfo?.(ctx.info);

  const data = generateMatVecData(rows, cols);
  const harness = { targetMs, warmups, runs };
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

  const dtypeAndLayoutBenchmarks: Array<[string, () => Promise<BenchmarkResult>]> = [
    ['f32-scalar', () => benchmarkF32Scalar(ctx, data, harness)],
    ['f32-vec4', () => benchmarkF32Vec4(ctx, data, harness)],
    ['f32-vec4-unrolled', () => benchmarkF32Vec4Unrolled(ctx, data, harness)],
    ['f32-vec4-shared', () => benchmarkF32Vec4Shared(ctx, data, 128, harness)],
    ['f16-scalar', () => benchmarkF16Scalar(ctx, data, harness)],
    ['f16-vec4', () => benchmarkF16Vec4(ctx, data, harness)],
    ['i8-packed', () => benchmarkI8Packed(ctx, data, harness)],
  ];

  for (const [id, run] of dtypeAndLayoutBenchmarks) {
    try {
      yield await run();
    } catch (error) {
      yield errorResult(id, id, '', 'dtype', rows, cols, error);
    }
  }

  const flopsBenchmarks: Array<[string, () => Promise<BenchmarkResult>]> = [
    ['flops-f32-scalar', () => benchmarkFlopsF32Scalar(ctx, flopsHarness)],
    ['flops-f32-vec4', () => benchmarkFlopsF32Vec4(ctx, flopsHarness)],
    ['flops-f32-mat4', () => benchmarkFlopsF32Mat4(ctx, flopsHarness)],
    ['flops-f16-scalar', () => benchmarkFlopsF16Scalar(ctx, flopsHarness)],
    ['flops-f16-vec4', () => benchmarkFlopsF16Vec4(ctx, flopsHarness)],
    ['flops-f16-mat4', () => benchmarkFlopsF16Mat4(ctx, flopsHarness)],
    ['flops-i8-scalar', () => benchmarkFlopsI8Scalar(ctx, flopsHarness)],
    ['flops-i8-vec4', () => benchmarkFlopsI8Vec4(ctx, flopsHarness)],
    ['flops-i8-mat4', () => benchmarkFlopsI8Mat4(ctx, flopsHarness)],
    ['flops-i8-dp4a', () => benchmarkFlopsI8Dp4a(ctx, flopsHarness)],
  ];

  for (const [id, run] of flopsBenchmarks) {
    try {
      yield await run();
    } catch (error) {
      yield errorResult(id, id, '', 'compute', rows, cols, error);
    }
  }

  try {
    for await (const result of runWorkgroupSweep(ctx, data, harness)) {
      // Skip re-yielding the wg=128 case a second time (already reported above).
      if (result.id === 'f32-vec4-shared-wg128') continue;
      yield result;
    }
  } catch (error) {
    yield errorResult('workgroup-sweep', 'Workgroup size sweep', '', 'workgroup', rows, cols, error);
  }

  try {
    yield await benchmarkMlpMultiLayer(ctx, { inputCols: cols, layerRows: mlpLayers, wgSize: 128, ...harness });
  } catch (error) {
    yield errorResult('mlp-multilayer', 'MLP multi-layer', '', 'multilayer', rows, cols, error);
  }
}
