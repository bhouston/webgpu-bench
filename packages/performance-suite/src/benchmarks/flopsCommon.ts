import type { GpuContext } from '../gpu/context.ts';
import { createUniformBuffer, createEmptyStorageBuffer } from '../gpu/buffers.ts';
import { createPipeline, runKernelBenchmark, skippedResult, type HarnessConfig } from './common.ts';
import type { BenchmarkResult } from '../types.ts';

export interface FlopsKernelSpec {
  id: string;
  label: string;
  description: string;
  wgsl: string;
  /** FLOPs (or integer ops, same MAC-as-2 convention) performed per loop iteration by a single thread. */
  flopsPerIteration: number;
  /**
   * Loop trip count to use when the harness doesn't override it. Kernels
   * that do more work per iteration (unrolled chains, matvec tiles) use
   * fewer trips so every kernel's dispatch lands in the same several-ms
   * range — long enough that per-thread setup, the final store, and
   * dispatch overhead are negligible next to the ALU work being measured.
   */
  defaultIterations?: number;
  /** Only raw-FLOPS kernels using f16 need this; skips with an explanation if the device lacks `shader-f16`. */
  requiresF16?: boolean;
  /** Only the `dot4I8Packed` kernel needs this; skips with an explanation if the device lacks `packed_4x8_integer_dot_product`. */
  requiresI8Dot?: boolean;
}

export interface FlopsHarnessConfig extends HarnessConfig {
  threads?: number;
  iterations?: number;
}

const DEFAULT_THREADS = 1024 * 1024;
const DEFAULT_ITERATIONS = 1024;

/**
 * Shared runner for the raw-FLOPS "little read/write, lots of ALU" compute
 * benchmarks: every kernel here shares the same `Params{threads,iterations}`
 * uniform + single scalar-per-thread output buffer layout, so only the WGSL
 * and the FLOPs-per-iteration accounting differ between them.
 */
export async function runFlopsBenchmark(
  ctx: GpuContext,
  spec: FlopsKernelSpec,
  harness: FlopsHarnessConfig = {},
): Promise<BenchmarkResult> {
  const threads = harness.threads ?? DEFAULT_THREADS;
  const iterations = harness.iterations ?? spec.defaultIterations ?? DEFAULT_ITERATIONS;

  if (spec.requiresF16 && !ctx.info.supportsF16) {
    return skippedResult(
      spec.id,
      spec.label,
      spec.description,
      'compute',
      threads,
      iterations,
      'Device/browser does not support the "shader-f16" WebGPU feature.',
    );
  }
  if (spec.requiresI8Dot && !ctx.info.supportsI8Dot) {
    return skippedResult(
      spec.id,
      spec.label,
      spec.description,
      'compute',
      threads,
      iterations,
      'Device/browser does not support the "packed_4x8_integer_dot_product" WebGPU feature.',
    );
  }

  const { device } = ctx;
  const pipeline = await createPipeline(device, spec.id, spec.wgsl);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([threads, iterations]), 'params');
  const outBuf = createEmptyStorageBuffer(device, threads * 4, 'out');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: outBuf } },
    ],
  });

  return runKernelBenchmark({
    id: spec.id,
    label: spec.label,
    description: spec.description,
    category: 'compute',
    ctx,
    rows: threads,
    cols: iterations,
    // Only the final per-thread scalar is ever written; the uniform read is negligible.
    bytes: threads * 4,
    flops: threads * iterations * spec.flopsPerIteration,
    workgroupsPerIteration: [Math.ceil(threads / 64), 1, 1],
    pipeline,
    bindGroup,
    targetMs: harness.targetMs,
    warmups: harness.warmups,
    minRuns: harness.minRuns,
    maxRuns: harness.maxRuns,
    precision: harness.precision,
  });
}
