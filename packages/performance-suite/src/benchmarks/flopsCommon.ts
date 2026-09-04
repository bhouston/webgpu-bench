import type { GpuContext } from '../gpu/context.ts';
import { createUniformBuffer, createEmptyStorageBuffer } from '../gpu/buffers.ts';
import type { WorkKnob } from '../gpu/benchmarkRunner.ts';
import {
  createPipeline,
  prepareKernelBenchmark,
  skippedResult,
  type BenchmarkMeta,
  type HarnessConfig,
  type PreparedBenchmark,
} from './common.ts';

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
  /**
   * Smallest loop trip count calibration may settle on. Also the first,
   * deliberately cheap probe a fresh device sees. Default 16.
   */
  minIterations?: number;
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
const DEFAULT_MIN_ITERATIONS = 16;

/**
 * Shared runner for the raw-FLOPS "little read/write, lots of ALU" compute
 * benchmarks: every kernel here shares the same `Params{threads,iterations}`
 * uniform + single scalar-per-thread output buffer layout, so only the WGSL
 * and the FLOPs-per-iteration accounting differ between them.
 *
 * The loop trip count is exposed to the sampler as a work knob: unless
 * `harness.iterations` pins it, the sampler calibrates it (between
 * `minIterations` and the kernel's default) so a single dispatch takes about
 * `targetDispatchMs` on this particular GPU — a fixed count that's a few ms
 * on a desktop is a display-freezing second on a phone.
 */
export async function prepareFlopsBenchmark(
  ctx: GpuContext,
  spec: FlopsKernelSpec,
  harness: FlopsHarnessConfig = {},
): Promise<PreparedBenchmark> {
  const threads = harness.threads ?? DEFAULT_THREADS;
  const maxIterations = harness.iterations ?? spec.defaultIterations ?? DEFAULT_ITERATIONS;
  const metaAtWork = (iterations: number): BenchmarkMeta => ({
    id: spec.id,
    label: spec.label,
    description: spec.description,
    category: 'compute',
    rows: threads,
    cols: iterations,
    // Only the final per-thread scalar is ever written; the uniform read is negligible.
    bytes: threads * 4,
    flops: threads * iterations * spec.flopsPerIteration,
  });
  const meta = metaAtWork(maxIterations);

  if (spec.requiresF16 && !ctx.info.supportsF16) {
    return {
      kind: 'skipped',
      result: skippedResult(meta, 'Device/browser does not support the "shader-f16" WebGPU feature.'),
    };
  }
  if (spec.requiresI8Dot && !ctx.info.supportsI8Dot) {
    return {
      kind: 'skipped',
      result: skippedResult(
        meta,
        'Device/browser does not support the "packed_4x8_integer_dot_product" WebGPU feature.',
      ),
    };
  }

  const { device } = ctx;
  const pipeline = await createPipeline(device, spec.id, spec.wgsl);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([threads, maxIterations]), 'params');
  const outBuf = createEmptyStorageBuffer(device, threads * 4, 'out');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: outBuf } },
    ],
  });

  // An explicit `iterations` pins the trip count; otherwise let the sampler size it.
  const work: WorkKnob | undefined =
    harness.iterations === undefined
      ? {
          min: Math.min(maxIterations, spec.minIterations ?? DEFAULT_MIN_ITERATIONS),
          max: maxIterations,
          apply: (iterations) => device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([threads, iterations])),
        }
      : undefined;

  return prepareKernelBenchmark({
    ...meta,
    ctx,
    workgroupsPerIteration: [Math.ceil(threads / 64), 1, 1],
    pipeline,
    bindGroup,
    work,
    metaAtWork,
    targetMs: harness.targetMs,
    targetDispatchMs: harness.targetDispatchMs,
    warmups: harness.warmups,
    maxIterations: harness.maxIterations,
  });
}
