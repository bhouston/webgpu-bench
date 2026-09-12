import type { GpuContext } from '../gpu/context.ts';
import { prepareFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { flopsF32ScalarWgsl } from '../shaders/flopsF32Scalar.ts';
import { flopsF32Vec4Wgsl } from '../shaders/flopsF32Vec4.ts';
import { flopsF32Mat4Wgsl } from '../shaders/flopsF32Mat4.ts';
import { flopsF32MatvecWgsl } from '../shaders/flopsF32Matvec.ts';
import type { PreparedBenchmark } from './common.ts';

/** Raw fp32 FLOPS: eight independent scalar FMA chains per thread, unrolled 4x. */
export function prepareFlopsF32Scalar(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'f32-fma-scalar',
      wgsl: flopsF32ScalarWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
    },
    harness,
  );
}

/** fp32 vec4 FLOPS: eight independent vec4<f32> FMA chains, unrolled 4x (same shape as scalar). */
export function prepareFlopsF32Vec4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'f32-fma-vec4',
      wgsl: flopsF32Vec4Wgsl,
      flopsPerIteration: 256,
      defaultIterations: 256,
    },
    harness,
  );
}

/** fp32 mat4 FLOPS: x = m*x + c chained with a mat4x4<f32>. */
export function prepareFlopsF32Mat4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'f32-fma-mat4',
      wgsl: flopsF32Mat4Wgsl,
      flopsPerIteration: 32,
      defaultIterations: 1024,
    },
    harness,
  );
}

/** fp32 register-resident matvec tile: 4x8 weights in registers, two dot() calls per output row. */
export function prepareFlopsF32Matvec(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'f32-fma-matvec',
      wgsl: flopsF32MatvecWgsl,
      flopsPerIteration: 64,
      defaultIterations: 512,
    },
    harness,
  );
}
