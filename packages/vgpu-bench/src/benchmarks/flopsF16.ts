import type { GpuContext } from '../gpu/context.ts';
import { prepareFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { flopsF16ScalarWgsl } from '../shaders/flopsF16Scalar.ts';
import { flopsF16Vec4Wgsl } from '../shaders/flopsF16Vec4.ts';
import { flopsF16Mat4Wgsl } from '../shaders/flopsF16Mat4.ts';
import { flopsF16MatvecWgsl } from '../shaders/flopsF16Matvec.ts';
import type { PreparedBenchmark } from './common.ts';

/** Raw fp16 FLOPS: eight independent scalar f16 FMA chains per thread, unrolled 4x. */
export function prepareFlopsF16Scalar(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f16-scalar',
      wgsl: flopsF16ScalarWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
      requiresF16: true,
    },
    harness,
  );
}

/** fp16 vec4 FLOPS: eight independent vec4<f16> FMA chains, unrolled 4x (same shape as scalar). */
export function prepareFlopsF16Vec4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f16-vec4',
      wgsl: flopsF16Vec4Wgsl,
      flopsPerIteration: 256,
      defaultIterations: 256,
      requiresF16: true,
    },
    harness,
  );
}

/** fp16 mat4 FLOPS: x = m*x + c chained with a mat4x4<f16>. */
export function prepareFlopsF16Mat4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f16-mat4',
      wgsl: flopsF16Mat4Wgsl,
      flopsPerIteration: 32,
      defaultIterations: 1024,
      requiresF16: true,
    },
    harness,
  );
}

/** fp16 register-resident matvec tile: 4x8 f16 weights in registers, two dot() calls per output row. */
export function prepareFlopsF16Matvec(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-f16-matvec',
      wgsl: flopsF16MatvecWgsl,
      flopsPerIteration: 64,
      defaultIterations: 512,
      requiresF16: true,
    },
    harness,
  );
}
