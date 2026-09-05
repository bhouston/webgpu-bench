import type { GpuContext } from '../gpu/context.ts';
import { prepareFlopsBenchmark, type FlopsHarnessConfig } from './flopsCommon.ts';
import { flopsI8ScalarWgsl } from '../shaders/flopsI8Scalar.ts';
import { flopsI8Vec4Wgsl } from '../shaders/flopsI8Vec4.ts';
import { flopsI8Mat4Wgsl } from '../shaders/flopsI8Mat4.ts';
import { flopsI8Dp4aWgsl } from '../shaders/flopsI8Dp4a.ts';
import { flopsI8MatvecWgsl } from '../shaders/flopsI8Matvec.ts';
import { flopsI8MatvecDp4aWgsl } from '../shaders/flopsI8MatvecDp4a.ts';
import type { PreparedBenchmark } from './common.ts';

/** Raw int8-range FLOPS: eight independent scalar i32 multiply-add chains per thread, unrolled 4x. */
export function prepareFlopsI8Scalar(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-scalar',
      wgsl: flopsI8ScalarWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
    },
    harness,
  );
}

/** int8-range vec4 FLOPS: eight independent vec4<i32> multiply-add chains, unrolled 4x (same shape as scalar). */
export function prepareFlopsI8Vec4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-vec4',
      wgsl: flopsI8Vec4Wgsl,
      flopsPerIteration: 256,
      defaultIterations: 256,
    },
    harness,
  );
}

/** int8-range mat4 FLOPS: a 4x4 integer matvec emulated via four dot(vec4<i32>) calls (WGSL has no mat4x4<i32>). */
export function prepareFlopsI8Mat4(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-mat4',
      wgsl: flopsI8Mat4Wgsl,
      flopsPerIteration: 32,
      defaultIterations: 1024,
    },
    harness,
  );
}

/** int8 register-resident matvec tile with weights unpacked to vec4<i32>, two integer dot() calls per output row. */
export function prepareFlopsI8Matvec(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-matvec',
      wgsl: flopsI8MatvecWgsl,
      flopsPerIteration: 64,
      defaultIterations: 512,
    },
    harness,
  );
}

/** int8 register-resident matvec tile on packed u32 words, two dot4I8Packed calls per output row. */
export function prepareFlopsI8MatvecDp4a(
  ctx: GpuContext,
  harness: FlopsHarnessConfig = {},
): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-matvec-dp4a',
      wgsl: flopsI8MatvecDp4aWgsl,
      flopsPerIteration: 64,
      defaultIterations: 512,
      requiresI8Dot: true,
    },
    harness,
  );
}

/** int8 packed-dot-product FLOPS: dot4I8Packed from the packed_4x8_integer_dot_product extension, run in a tight accumulation loop over eight independent accumulators. */
export function prepareFlopsI8Dp4a(ctx: GpuContext, harness: FlopsHarnessConfig = {}): Promise<PreparedBenchmark> {
  return prepareFlopsBenchmark(
    ctx,
    {
      id: 'flops-i8-dp4a',
      wgsl: flopsI8Dp4aWgsl,
      flopsPerIteration: 64,
      defaultIterations: 256,
      requiresI8Dot: true,
    },
    harness,
  );
}
